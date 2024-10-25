"""Python entrypoint of weight conversion."""

import dataclasses
import math
import os
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

from tvm import tir
from tvm.contrib import tvmjs
from tvm.runtime import DataType, Device, NDArray
from tvm.runtime import cpu as cpu_device
from tvm.target import Target

from mlc_llm.loader import LOADER
from mlc_llm.model import Model
from mlc_llm.quantization import Quantization
from mlc_llm.support import logging, tqdm
from mlc_llm.support.preshard import apply_preshard
from mlc_llm.support.style import bold, green

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ConversionArgs:  # pylint: disable=too-many-instance-attributes
    """Arguments to MLC LLM's weight conversation and quantization flow."""

    config: Path
    quantization: Quantization
    model: Model
    device: Device
    source: Path
    source_format: str
    output: Path

    def display(self) -> None:
        """Display the arguments to stdout."""

        def _device_to_str(device: Device) -> str:
            return f"{Device.MASK2STR[device.device_type]}:{device.device_id}"

        out = StringIO()
        print(f"{bold('Weight conversion with arguments:')}", file=out)
        print(f"  {bold('--config'):<25} {self.config}", file=out)
        print(f"  {bold('--quantization'):<25} {self.quantization}", file=out)
        print(f"  {bold('--model-type'):<25} {self.model.name}", file=out)
        print(f"  {bold('--device'):<25} {_device_to_str(self.device)}", file=out)
        print(f"  {bold('--source'):<25} {self.source}", file=out)
        print(f"  {bold('--source-format'):<25} {self.source_format}", file=out)
        print(f"  {bold('--output'):<25} {self.output}", file=out)
        print(out.getvalue().rstrip())

# 模型格式转换函数
# 
def _convert_args(args: ConversionArgs) -> None:  # pylint: disable=too-many-locals
    # pre_shards 预分片，主要用于多GPU
    pre_shards_num = os.getenv("MLC_INTERNAL_PRESHARD_NUM")
    # model config & quantization config
    model_config = args.model.config.from_file(args.config)
    if (
        args.quantization.kind == "ft-quant"
        and hasattr(model_config, "tensor_parallel_shards")
        and model_config.tensor_parallel_shards > 1
    ):
        raise NotImplementedError
    if pre_shards_num is not None:
        model_config.tensor_parallel_shards = int(pre_shards_num)
    # model.quantize 是对应MODEL类所支持的量化类型 (model\model.py#270)，及其对应量化接口的映射表
    # 如qwen2中，按分组量化进行
    # quantize={
    #     "no-quant": qwen2_quantization.no_quant,
    #     "group-quant": qwen2_quantization.group_quant,
    #     "ft-quant": qwen2_quantization.ft_quant,
    # },
    #
    # args.model.quantize[args.quantization.kind] 通过 args.quantization.kind("group-quant") 转为对应量化函数（“qwen2_quantization.group_quant”）
    # 对应 model\qwen2\qwen2_quantization.py#14，里面是该模型的分组量化操作接口。通过输入参数, 在qwen2_quantization.group_quant里调用 
    # GroupQuantize（quantization\group_quantization.py#28）的 quantize_model 函数，完成从非量化模型到量化模型的转换。
    #
    # 输出得到量化后的模型 model (nn.Module) 和量化表 quantize_map (QuantizeMapping)
    # 注意，此时的输出仅是完成从非量化模型到量化模型的转换，实际权重并未进行量化，需要根据量化表，在下面加载权重的时候再进行权重的量化。
    #      量化表里包含有每个node的量化情况，可量化节点有nn.Linear / nn.Embedding / MixtralExperts三种，会对这些node进行类型的替换，并设定量化函数。
    #      (quantization\group_quantization.py#110)
    model, quantize_map = args.model.quantize[args.quantization.kind](
        model_config, args.quantization
    )
    # args.model是MODEL，MODEL.model是qwen2_model.QWen2LMHeadModel，是nn.Module类型，在tvm中处于前端位置。
    # 经过量化后得到的 model 也是nn.Module类型，里面的参数已被量化所修改(有权重的一些信息, 但不含权重数据)。
    # export_tvm (relax\frontend\nn\core.py#447)，将 nn.Module 转为 TVM IRModule和参数。
    # IRModule: tvm一系列的优化策略都是基于IRModule进行的. 
    # params: 对应着模型权重的一些参数.
    # ext_mods: 在模型中被使用的其他模块.
    # 这里只用到了params, 用于检查参数type/shape等情况, 防止量化模型的转换出问题?
    # get_default_spec (model\qwen2\qwen2_model.py#343) 是 指定模型的nn.Module的 每个输入名称映射到规范的字典，它定义了输入形状和dtype。
    # 模型导出时，需要根据这份规范字典来导出。
    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),  # type: ignore[attr-defined]
        allow_extern=True,
    )
    named_params = dict(_named_params)

    if pre_shards_num is not None:
        # 如果是多GPU并行，则将模型参数分发到多个GPU上
        named_params, preshard_funcs = apply_preshard(named_params, int(pre_shards_num), args)
    else:
        preshard_funcs = None

    # 检查参数的name / shape 和 type。
    def _check_param(name: str, param: NDArray):
        nonlocal named_params
        if name not in named_params:
            raise ValueError(f"Parameter not found in model: {name}")
        if name in param_names:
            raise ValueError(f"Duplication: Parameter {name} already computed")

        # Check shape (possibly dynamic)
        def _check_shape(actual: tuple, expect: tuple):  # expect can have tir.Var
            if len(actual) != len(expect):
                return False
            for actual_i, expect_i in zip(actual, expect):
                assert isinstance(expect_i, (int, tir.Var))
                if isinstance(expect_i, int) and actual_i != expect_i:
                    return False
            return True

        expect_shape = named_params[name].shape
        actual_shape = param.shape
        if not _check_shape(actual_shape, expect_shape):
            raise ValueError(
                f"Parameter {name} has shape {param.shape}, but expected {expect_shape}"
            )
        # Check dtype
        actual_dtype = param.dtype
        expect_dtype = named_params[name].dtype
        if actual_dtype != expect_dtype:
            raise ValueError(
                f"Parameter {name} has dtype {param.dtype}, but expected {expect_dtype}"
            )
        del named_params[name]

    # load and quantize
    param_names = set()
    total_bytes = 0.0
    total_params: int

    def _param_generator() -> Iterator[Tuple[str, NDArray]]:
        nonlocal total_params, total_bytes
        # Target.from_device检测对应device是否有效
        with Target.from_device(args.device), tqdm.redirect():
            # args.source_format 表示权重的格式（loader\loader.py#9）
            # 有 huggingface-torch / huggingface-safetensor / awq 三种格式，但都对应着HuggingFaceLoader一个类。
            # model.source 对应 model文件夹中特定模型的loader.py的huggingface函数（model\qwen2\qwen2_loader.py#16），用于得到
            loader = LOADER[args.source_format](
                path=args.source,
                extern_param_map=args.model.source[args.source_format](
                    model_config, args.quantization
                ),
                quantize_param_map=quantize_map,
            )
            # HuggingFaceLoader会加载和解析HuggingFace格式，
            # 在加载权重的同时，根据上面得到的量化模型结构和量化表，将需要量化的节点的参数进行量化。（loader\huggingface_loader.py#121）
            # 并将其转换为mlc-llm的格式，函数返回参数及其名字
            for name, param in loader.load(device=args.device, preshard_funcs=preshard_funcs):
                _check_param(name, param)
                param_names.add(name)
                param = param.copyto(cpu_device()) # CJM_TODO: WHY？
                # math.prod()是一个用于计算可迭代对象中所有元素乘积的函数
                total_bytes += math.prod(param.shape) * DataType(param.dtype).itemsize()
                yield name, param
        total_params = loader.stats.total_param_num

    def _metadata_callback() -> Dict[str, Any]:
        return {
            "ParamSize": len(param_names),
            "ParamBytes": total_bytes,
            "BitsPerParam": total_bytes * 8.0 / total_params,
        }

    # dump to output directory
    tvmjs.dump_ndarray_cache(
        _param_generator(),
        str(args.output),
        meta_data=_metadata_callback,
        encode_format="f32-to-bf16",
        show_progress=False,
    )
    if named_params:
        raise ValueError(f"Parameter not found in source: {', '.join(named_params.keys())}")
    # Log necessary statistics
    logger.info(
        "%s after quantization: %.3f GB",
        green("Parameter size"),
        total_bytes / (1024**3),
    )
    logger.info(f"%s: {total_params:,}", green("Total parameters"))
    logger.info(
        "%s: %.3f",
        green("Bits per parameter"),
        total_bytes * 8.0 / total_params,
    )
    logger.info("Saved to directory: %s", bold(str(args.output)))


def convert_weight(  # pylint: disable=too-many-arguments
    config: Path,
    quantization: Quantization,
    model: Model,
    device: Device,
    source: Path,
    source_format: str,
    output: Path,
):
    """MLC LLM's weight conversation and quantization flow."""
    args = ConversionArgs(config, quantization, model, device, source, source_format, output)
    args.display()
    _convert_args(args)
