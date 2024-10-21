"""Common utilities for quantization"""

from typing import Callable, List, Optional, Sequence

from tvm import IRModule
from tvm import dlight as dl
from tvm import relax, te, tir
from tvm.relax.frontend import nn
from tvm.runtime import DataType, DataTypeCode
from tvm.target import Target

from mlc_llm.support import tensor_parallel as tp


def convert_uint_to_float(  # pylint: disable=too-many-arguments
    weight: te.Tensor,
    bits: int,
    num_elem_per_storage: int,
    storage_dtype: str,
    model_dtype: str,
    axis: int = -1,
    out_shape: Optional[List[tir.PrimExpr]] = None,
    ft_reorder: Optional[bool] = False,
) -> te.Tensor:
    """Convert a quantized uint weight to an unquantized float weight."""
    # 如int4，1111=1<<4-1, 按storage_dtype为uint32算，则tir_bin_mask为00000000 00000000 00000000 00001111
    tir_bin_mask = tir.const((1 << bits) - 1, storage_dtype)
    if out_shape is None:
        out_shape = weight.shape                # 此时权重维度是基于storage_type的uint32来算的，对于int4，num_elem_per_storage为8.
        out_shape[axis] *= num_elem_per_storage # Linear层传的axis就是量化的axis，其他量化层是-1
    axis = axis if axis >= 0 else len(out_shape) + axis
    # 按out_shape进行索引，weight是uint32存储，则每次取出一个uint32，以idx[axis] // num_elem_per_storage，即同一个uint32会被重复取8次。
    # idx[axis] % num_elem_per_storage是从0到7，
    # 0(0+0), 1(4+0), 2(0+1), 3(4+1), 4(0+2), 5(4+2), 6(0+3), 7(4,3)
    #      -> 0-0,  1-4, 2-1,  3-5, 4-2,  5-6,  6-3, 7-7  
    # *bit -> 0-0, 1-16, 2-4, 3-20, 4-8, 5-24, 6-12, 7-28, 
    # 后面的数字是位移的偏移量，将对应4位移动到最右边，与tir_bin_mask做位与运算。
    return te.compute(
        shape=out_shape,
        fcompute=lambda *idx: tir.bitwise_and(
            tir.shift_right(
                weight(*idx[:axis], idx[axis] // num_elem_per_storage, *idx[axis + 1 :]),
                (
                    (
                        (idx[axis] % num_elem_per_storage) % 2 * 4
                        + (idx[axis] % num_elem_per_storage) // 2
                    )
                    * bits
                    if ft_reorder
                    else (idx[axis] % num_elem_per_storage) * bits
                ).astype(storage_dtype),
            ),
            tir_bin_mask,
        ).astype(model_dtype),
    )


def is_final_fc(name: str) -> bool:
    """Determines whether the parameter is the last layer based on its name."""
    # TODO: use more specious condition to determine final fc  # pylint: disable=fixme
    return name in ["head", "lm_head", "lm_head.linear", "embed_out"]


def is_moe_gate(name: str, node: nn.Linear) -> bool:
    """Check whether the parameter is the MoE gate layer."""
    return name.endswith("gate") and isinstance(node.out_features, int) and node.out_features <= 64

# 针对指定设备，执行一些常规transform优化，编译量化函数。
def compile_quantize_func(mod: IRModule, device) -> Callable:
    """Compile a quantization function for a given device."""
    device_type = device.MASK2STR[device.device_type]
    if device_type in ["cuda", "rocm", "metal", "vulkan", "opencl"]:
        target = Target.current()
        if target is None:
            target = Target.from_device(device)
        with target:
            mod = dl.ApplyDefaultSchedule(  # type: ignore   # pylint: disable=not-callable
                dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(),
                dl.gpu.Fallback(),
            )(mod)
    elif device_type == "cpu":
        target = "llvm"
        mod = relax.transform.LegalizeOps()(mod)
    else:
        raise NotImplementedError(f"Device type {device_type} is not supported")
    ex = relax.build(mod, target=target)
    vm = relax.VirtualMachine(ex, device)  # pylint: disable=invalid-name
    return vm["main"]


def apply_sharding(shard_strategy, name: str, weight: nn.Parameter):
    """Apply sharding strategy to a weight."""
    if isinstance(shard_strategy, tp.ShardSingleDim):
        weight.attrs["shard_strategy"] = tp.ShardSingleDim(
            name=name,
            dim=shard_strategy.dim,
            segs=shard_strategy.segs,
        )
    else:
        raise NotImplementedError(f"Unknowing sharding strategy: {shard_strategy}")


def convert_uint_packed_fp8_to_float(  # pylint: disable=too-many-arguments
    weight: te.Tensor,
    num_elem_per_storage: int,
    storage_dtype: str,
    model_dtype: str,
    quant_dtype: str,
    axis: int = -1,
    out_shape: Optional[Sequence[tir.PrimExpr]] = None,
) -> te.Tensor:
    """Unpack a fp8 value from the storage dtype and convert to float."""
    assert quant_dtype in ["e4m3_float8", "e5m2_float8"]
    assert DataType(storage_dtype).type_code == DataTypeCode.UINT
    bits = DataType(quant_dtype).bits
    elem_storage_dtype = DataType(f"uint{bits}")
    tir_bin_mask = tir.const((1 << bits) - 1, "uint8")
    if axis < 0:
        axis += len(weight.shape)
    if out_shape is None:
        out_shape = (
            *weight.shape[:axis],
            weight.shape[axis] * num_elem_per_storage,
            *weight.shape[axis + 1 :],
        )
    axis = axis if axis >= 0 else len(out_shape) + axis
    return te.compute(
        shape=out_shape,
        fcompute=lambda *idx: tir.reinterpret(
            quant_dtype,
            tir.bitwise_and(
                tir.shift_right(
                    weight(*idx[:axis], idx[axis] // num_elem_per_storage, *idx[axis + 1 :]),
                    ((idx[axis] % num_elem_per_storage) * bits).astype(storage_dtype),
                ).astype(elem_storage_dtype),
                tir_bin_mask,
            ),
        ).astype(model_dtype),
    )


def pack_weight(
    weight: te.Tensor,
    axis: int,
    num_elem_per_storage: int,
    weight_dtype: str,
    storage_dtype: str,
    out_shape: Optional[Sequence[tir.PrimExpr]] = None,
):  # pylint: disable=too-many-arguments
    """Convert a tensor to a packed format by packing consecutive bits.
    This can be useful for sub-byte quantization.

    Parameters
    ----------
    weight : te.Tensor
        The weight
    axis : int
        The axis to pack.
    num_elem_per_storage : int
        The number of elements per storage.
    weight_dtype : str
        The dtype of the input tensor.
    storage_dtype : str
        The dtype of the packed tensor.
    out_shape : Optional[Sequence[tir.PrimExpr]]
        The output shape of the packed tensor. Zero-padding is added if needed.
    """
    assert weight.dtype == storage_dtype
    shape = weight.shape
    if axis < 0:
        axis += len(shape)
    k = shape[axis]
    axis = axis if axis >= 0 else len(shape) + axis
    if out_shape is None:
        out_shape = (*shape[:axis], tir.ceildiv(k, num_elem_per_storage), *shape[axis + 1 :])
    r = te.reduce_axis((0, num_elem_per_storage), name="r")  # pylint: disable=invalid-name
    packed_weight = te.compute(
        shape=out_shape,
        fcompute=lambda *idx: tir.sum(
            tir.if_then_else(
                idx[axis] * num_elem_per_storage + r < k,
                weight(*idx[:axis], idx[axis] * num_elem_per_storage + r, *idx[axis + 1 :])
                << (r * DataType(weight_dtype).bits),
                tir.const(0, storage_dtype),
            ),
            axis=r,
        ),
        name="packed_weight",
    ).astype(storage_dtype)
    return packed_weight
