"""The pass that attaches embedding allocation function to the IRModule."""

from typing import Any, Dict

import tvm
from tvm import IRModule, relax

# 将embeding张量分配的relas函数附加到IRModule中. "embed"在qwen2中只有一个
@tvm.transform.module_pass(opt_level=0, name="AttachAllocEmbeddingTensorFunc")
class AttachAllocEmbeddingTensorFunc:  # pylint: disable=too-few-public-methods
    """Attach embedding tensor allocation Relax function to IRModule."""

    def __init__(self, metadata: Dict[str, Any]):
        self.metadata = metadata

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """Entrypoint"""
        embed_func = None
        for gv, func in mod.functions_items():
            if gv.name_hint == "embed":
                embed_func = func

        if embed_func is None:
            return mod

        hidden_size = embed_func.ret_struct_info.shape[-1]
        dtype = embed_func.ret_struct_info.dtype
        # 基于原来的IRModule全量构建Builder, 然后添加一个分配tensor内存的函数进去
        bb = relax.BlockBuilder(mod)
        with bb.function("alloc_embedding_tensor", []):
            bb.emit_func_output(
                bb.emit(
                    relax.op.builtin.alloc_tensor(
                        relax.ShapeExpr([self.metadata["prefill_chunk_size"], hidden_size]),
                        dtype,
                        runtime_device_index=0,
                    )
                )
            )
        return bb.finalize()
