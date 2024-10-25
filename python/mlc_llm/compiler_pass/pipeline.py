"""The compilation pipeline for LLM applications."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import tvm
from tvm import IRModule
from tvm import dlight as dl
from tvm.relax import register_pipeline  # pylint: disable=no-name-in-module
from tvm.relax.frontend import nn

from mlc_llm.interface.compiler_flags import IPCAllReduceStrategyType
from mlc_llm.support import logging

from .attach_embedding_allocator import AttachAllocEmbeddingTensorFunc
from .attach_logit_processor import AttachLogitProcessFunc
from .attach_sampler import AttachGPUSamplingFunc
from .attach_softmax_with_temperature import AttachSoftmaxWithTemperature
from .attach_spec_decode_aux_funcs import AttachSpecDecodeAuxFuncs
from .attach_support_info import (
    AttachAdditionalPrimFuncs,
    AttachCUDAGraphSymbolicCaptureHints,
    AttachMemoryPlanAttr,
    AttachPipelineParallelStages,
    AttachVariableBounds,
)
from .blas_dispatch import BLASDispatch
from .clean_up_tir_attrs import CleanUpTIRAttrs
from .dispatch_kv_cache_creation import DispatchKVCacheCreation
from .estimate_memory_usage import AttachMetadataWithMemoryUsage
from .fuse_add_norm import FuseAddRMSNorm
from .fuse_dequantize_matmul_ewise import FuseDequantizeMatmulEwise
from .fuse_dequantize_take import FuseDequantizeTake
from .fuse_dequantize_transpose import FuseDequantizeTranspose
from .fuse_ft_dequantize_matmul_epilogue import FuseFTDequantizeEpilogue
from .fuse_transpose_matmul import FuseTransposeMatmul
from .lift_global_buffer_alloc import LiftTIRGlobalBufferAlloc
from .low_batch_specialization import LowBatchGemvSpecialize
from .pipeline_parallel_rewrite import PipelineParallelRewrite
from .scatter_tuple_get_item import ScatterTupleGetItem

logger = logging.getLogger(__name__)


@tvm.transform.module_pass(opt_level=0, name="_LogProgress")
class _LogProgress:  # pylint: disable=too-few-public-methods
    """A dummy compiler pass that does nothing but logging."""

    def __init__(self, *args):
        self.args = args

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """A dummy transformation"""
        logger.info(*self.args)
        return mod


@tvm.transform.module_pass(opt_level=0, name="DebugDump")
class _DebugDump:  # pylint: disable=too-few-public-methods
    """A dummy compiler pass that does nothing but logging.
    Only enabled when debug_dump is not None"""

    def __init__(self, file_name: str, file_path: Optional[Path], show_meta: bool = False):
        self.file_name = file_name
        self.file_path = file_path
        self.show_meta = show_meta

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """A dummy transformation that dumps the module to file"""
        if self.file_path is not None:
            # NOTE: We use debug level here to avoid spamming the console
            logger.debug("Dumping IR to %s", self.file_path / self.file_name)
            with open(self.file_path / self.file_name, "w", encoding="utf-8") as f:
                f.write(mod.script(show_meta=self.show_meta))
        return mod


@register_pipeline("mlc_llm")
def _mlc_llm_pipeline(  # pylint: disable=too-many-arguments
    target: tvm.target.Target,
    flashinfer: bool = False,
    cublas_gemm: bool = False,
    faster_transformer: bool = False,  # pylint: disable=unused-argument
    allreduce_strategy: IPCAllReduceStrategyType = IPCAllReduceStrategyType.NONE,
    variable_bounds: Dict[str, int] = None,
    cuda_graph_symbolic_capture_hints: Dict[str, List[str]] = None,
    additional_tirs: Dict[str, tvm.tir.PrimFunc] = None,
    metadata: Dict[str, Any] = None,
    ext_mods: List[nn.ExternModule] = None,
    debug_dump: Optional[Path] = None,
):
    variable_bounds = variable_bounds or {}
    cuda_graph_symbolic_capture_hints = cuda_graph_symbolic_capture_hints or {}
    additional_tirs = additional_tirs or {}
    metadata = metadata or {}
    ext_mods = ext_mods or []
    tensor_parallel_shards = metadata.get("tensor_parallel_shards", 1)

    # tvm.transform.Sequential将所需的compile_passl类构造的对象都打包在一起,作为一个完整的编译pipeline, 
    # 输入需要编译的IRModule, 依次经过打包的这些compile_pass类的transform_module方法, 得到转换后的IRModule. 
    #
    # 打包的compile_pass都会有这样的修饰器 @tvm.transform.module_pass(opt_level=0, name="DispatchKVCacheCreation"), 
    # 修饰器的位置在 module_pass (3rdparty\tvm\python\tvm\ir\transform.py#325)
    # compile_pass里面会定义有 transform_module 方法, 是执行编译优化的主体函数. CJM_TODO: transform_module在哪里被调用,如何跟c++关联?
    #
    # CJM_TODO: 每一个compile_pass的含义.
    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        seq = tvm.transform.Sequential(
            [
                #####
                # Phase 0. Add additional information for compilation and remove unused Relax func
                # 将IRModule中的函数"create_paged_kv_cache"变成了"create_tir_paged_kv_cache" 和 "create_flashinfer_paged_kv_cache" 两个函数
                DispatchKVCacheCreation(target, flashinfer, metadata),
                # 为了保证数值稳定性，如何处理不同的温度参数情况，并通过内核代码实现这些逻辑。
                AttachSoftmaxWithTemperature(target),
                # 附加变量的上下界信息, 便于优化
                AttachVariableBounds(variable_bounds),
                # 设置cuda graph需要处理的提示符. 主要是几个batch操作, 加入如batch_size等重要信息,使cuda graph能针对性处理.
                AttachCUDAGraphSymbolicCaptureHints(cuda_graph_symbolic_capture_hints),
                # 将流水线并行所需要的arallel_shards绑定到对应函数中
                AttachPipelineParallelStages(metadata["pipeline_parallel_stages"]),
                # 将logit处理用的TIR functions转为IRModule.
                AttachLogitProcessFunc(target),
                AttachAdditionalPrimFuncs(additional_tirs),
                # 将embeding张量分配的relas函数附加到IRModule中.
                AttachAllocEmbeddingTensorFunc(metadata),
                # 把GPU采样函数添加到IRModule中.
                AttachGPUSamplingFunc(target, variable_bounds),
                AttachSpecDecodeAuxFuncs(tensor_parallel_shards),
                AttachMemoryPlanAttr(),
                tvm.tir.transform.BindTarget(tvm.target.Target.current(allow_none=False)),
                _DebugDump("debug-phase0.py", debug_dump, show_meta=False),
                # Phase 1. Passes on high-level operator graph
                _LogProgress("Running TVM Relax graph-level optimizations"),
                FuseFTDequantizeEpilogue(),
                FuseDequantizeTranspose(),
                BLASDispatch(target) if cublas_gemm else tvm.transform.Sequential([]),
                FuseAddRMSNorm(target=target),
                FuseTransposeMatmul(),
                _DebugDump("debug-phase1.py", debug_dump, show_meta=False),
                # Phase 2. Lowering to TIR, inherited TVM Relax's official "zero" pipeline
                _LogProgress("Lowering to TVM TIR kernels"),
                tvm.relax.backend.DispatchSampling(),
                tvm.relax.backend.DispatchSortScan(),
                tvm.relax.transform.LegalizeOps(),
                tvm.relax.transform.AnnotateTIROpPattern(),
                tvm.relax.transform.FoldConstant(),
                tvm.relax.transform.FuseOps(),
                tvm.relax.transform.FuseTIR(),
                _DebugDump("debug-phase2.py", debug_dump, show_meta=False),
                # Phase 3. Passes on TIR
                _LogProgress("Running TVM TIR-level optimizations"),
                FuseDequantizeMatmulEwise(),
                FuseDequantizeTake(),
                tvm.relax.transform.DeadCodeElimination(),
                CleanUpTIRAttrs(["op_pattern"]),
                _DebugDump("debug-phase3.py", debug_dump, show_meta=False),
                # Phase 4. Low-level Optimizations
                _LogProgress("Running TVM Dlight low-level optimizations"),
                LowBatchGemvSpecialize(),
                dl.ApplyDefaultSchedule(
                    dl.gpu.Matmul(),
                    dl.gpu.GEMV(),
                    dl.gpu.Reduction(),
                    dl.gpu.GeneralReduction(),
                    dl.gpu.Fallback(),
                ),
                _DebugDump("debug-phase4.py", debug_dump, show_meta=False),
                _LogProgress("Lowering to VM bytecode"),
                LiftTIRGlobalBufferAlloc(),
                (
                    tvm.tir.transform.ForceNarrowIndexToInt32()
                    if target.kind.name != "cuda"
                    else tvm.transform.Sequential([])
                ),
                ScatterTupleGetItem(),
                PipelineParallelRewrite(),
                _DebugDump("after-pipeline-rewrite.py", debug_dump, show_meta=False),
                tvm.relax.transform.RewriteDataflowReshape(),
                tvm.relax.transform.ToNonDataflow(),
                tvm.relax.transform.RemovePurityChecking(),
                tvm.relax.transform.CallTIRRewrite(),
                (
                    tvm.relax.transform.IPCAllReduceRewrite(allreduce_strategy)
                    if allreduce_strategy != IPCAllReduceStrategyType.NONE
                    else tvm.transform.Sequential([])
                ),
                tvm.relax.transform.StaticPlanBlockMemory(),
                AttachMetadataWithMemoryUsage(metadata),
                tvm.relax.transform.RewriteCUDAGraph(),
                tvm.relax.transform.LowerGPUIPCAllocStorage(),
                tvm.relax.transform.LowerAllocTensor(),
                tvm.relax.transform.KillAfterLastUse(),
                tvm.relax.transform.LowerRuntimeBuiltin(),
                tvm.relax.transform.VMShapeLower(),
                tvm.relax.transform.AttachGlobalSymbol(),
                _DebugDump("debug-final.py", debug_dump, show_meta=False),
                _LogProgress("Compiling external modules"),
                tvm.relax.transform.AttachExternModules(ext_mods),
                _LogProgress("Compilation complete! Exporting to disk"),
            ]
        )
        mod = seq(mod)
        return mod

    return _pipeline
