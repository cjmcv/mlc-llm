"""A centralized registry of all existing quantization methods and their configurations."""

from typing import Any, Dict

from .awq_quantization import AWQQuantize
from .ft_quantization import FTQuantize
from .group_quantization import GroupQuantize
from .no_quantization import NoQuantize
from .per_tensor_quantization import PerTensorQuantize

Quantization = Any
"""Quantization is an object that represents an quantization algorithm. It is required to
have the following fields:

    name : str
        The name of the quantization algorithm, for example, "q4f16_1".

    kind : str
        The kind of quantization algorithm, for example, "group-quant", "faster-transformer".

It is also required to have the following method:

    def quantize_model(self, module: nn.Module) -> nn.Module:
        ...

    def quantize_weight(self, weight: tvm.runtime.NDArray) -> List[tvm.runtime.NDArray]:
        ...
"""

# GroupQuantize: 分组量化，将一个tensor划分为多个组，每个组可以独立地使用一组量化参数。
# AWQQuantize：Activation-aware Weight Quantization，激活感知量化，也可能会涉及到分组，但与传统分组量化不同。
# FTQuantize：FasterTransformer中的量化方式
# PerTensorQuantize：逐tensor量化，一个tensor用一组量化参数。
# 注意 per-channel量化可以看成是 分组的数量等同于通道数 的 分组量化的特殊版本。
#      即量化精度 per-tensor < groupwise, per-tensor < per-channel。
#      而groupwise和per-channel的颗粒度取决于分组数量，如分组数量小于通道数，则 groupwise < per-channel; 
#                                                   如分组数量大于通道数，则 groupwise > per-channel。分组数量是可以大于通道数的
#
# 每个量化类会包含有三个类，如 GroupQuantize 里有 GroupQuantizeLinear / GroupQuantizeEmbedding / GroupQuantizeMixtralExperts，
#                           由主类GroupQuantize进行统一调用。
#
# # CJM_TODO: 动态量化不需要保存scale，则也不需要校准数据集？？？？
QUANTIZATION: Dict[str, Quantization] = {
    "q0f16": NoQuantize(
        name="q0f16",
        kind="no-quant",
        model_dtype="float16",
    ),
    "q0f32": NoQuantize(
        name="q0f32",
        kind="no-quant",
        model_dtype="float32",
    ),
    "q3f16_0": GroupQuantize(
        name="q3f16_0",
        kind="group-quant",
        group_size=40,
        quantize_dtype="int3",
        storage_dtype="uint32",
        model_dtype="float16",
        linear_weight_layout="KN",
        quantize_embedding=True,
        quantize_final_fc=True,
    ),
    "q3f16_1": GroupQuantize(
        name="q3f16_1",
        kind="group-quant",
        group_size=40,
        quantize_dtype="int3",
        storage_dtype="uint32",
        model_dtype="float16",
        linear_weight_layout="NK",
        quantize_embedding=True,
        quantize_final_fc=True,
    ),
    "q4f16_0": GroupQuantize(
        name="q4f16_0",
        kind="group-quant",
        group_size=32,
        quantize_dtype="int4",
        storage_dtype="uint32",
        model_dtype="float16",
        linear_weight_layout="KN",
        quantize_embedding=True,
        quantize_final_fc=True,
    ),
    "q4f16_1": GroupQuantize(
        name="q4f16_1",
        kind="group-quant",
        group_size=32,
        quantize_dtype="int4",
        storage_dtype="uint32",
        model_dtype="float16",
        linear_weight_layout="NK",
        quantize_embedding=True,
        quantize_final_fc=True,
    ),
    "q4f32_1": GroupQuantize(
        name="q4f32_1",
        kind="group-quant",
        group_size=32,
        quantize_dtype="int4",
        storage_dtype="uint32",
        model_dtype="float32",
        linear_weight_layout="NK",
        quantize_embedding=True,
        quantize_final_fc=True,
    ),
    "q4f16_2": GroupQuantize(
        name="q4f16_2",
        kind="group-quant",
        group_size=32,
        quantize_dtype="int4",
        storage_dtype="uint32",
        model_dtype="float16",
        linear_weight_layout="NK",
        quantize_embedding=False,
        quantize_final_fc=False,
    ),
    "q4f16_autoawq": AWQQuantize(
        name="q4f16_autoawq",
        kind="awq",
        group_size=128,
        quantize_dtype="int4",
        storage_dtype="uint32",
        model_dtype="float16",
    ),
    "q4f16_ft": FTQuantize(
        name="q4f16_ft",
        kind="ft-quant",
        quantize_dtype="int4",
        storage_dtype="int8",
        model_dtype="float16",
    ),
    "e5m2_e5m2_f16": PerTensorQuantize(
        name="e5m2_e5m2_f16",
        kind="per-tensor-quant",
        activation_dtype="e5m2_float8",
        weight_dtype="e5m2_float8",
        storage_dtype="e5m2_float8",
        model_dtype="float16",
        quantize_final_fc=False,
        quantize_embedding=False,
        quantize_linear=True,
        use_scale=False,
    ),
    "e4m3_e4m3_f16": PerTensorQuantize(
        name="e4m3_e4m3_f16",
        kind="per-tensor-quant",
        activation_dtype="e4m3_float8",
        weight_dtype="e4m3_float8",
        storage_dtype="e4m3_float8",
        model_dtype="float16",
        quantize_final_fc=False,
        quantize_embedding=False,
        quantize_linear=True,
        use_scale=True,
        calibration_mode="inference",
    ),
    "e4m3_e4m3_f16_max_calibrate": PerTensorQuantize(
        name="e4m3_e4m3_f16_max_calibrate",
        kind="per-tensor-quant",
        activation_dtype="e4m3_float8",
        weight_dtype="e4m3_float8",
        storage_dtype="e4m3_float8",
        model_dtype="float16",
        quantize_final_fc=False,
        quantize_embedding=False,
        quantize_linear=True,
        use_scale=True,
        calibration_mode="max",
    ),
}
