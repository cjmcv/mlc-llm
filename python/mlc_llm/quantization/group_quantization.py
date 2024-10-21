"""The group quantization config"""

from dataclasses import dataclass
from functools import partial
from typing import Any, List, Literal, Optional, Tuple, Union

from tvm import DataType, DataTypeCode, IRModule, relax, te, tir, topi
from tvm.relax.frontend import nn
from tvm.runtime import NDArray

from mlc_llm.loader import QuantizeMapping
from mlc_llm.nn import MixtralExperts
from mlc_llm.support import logging

from .utils import (
    apply_sharding,
    compile_quantize_func,
    convert_uint_to_float,
    is_final_fc,
    is_moe_gate,
    pack_weight,
)

logger = logging.getLogger(__name__)

## python语法
# 数据类：使用@dataclass装饰器后，Python 会自动为数据类生成一些特殊的方法，比如__init__方法、__repr__方法等
# typing.Literal是用于类型提示，限制参数的取值范围。
# __post_init__是一个特殊的方法，通常在数据类（使用@dataclass装饰的类）中使用，在对象初始化完成后被自动调用，
# functools.partial：它用于创建一个新的可调用对象，这个对象在调用时就像是原始函数被部分应用了某些参数一样。

@dataclass
class GroupQuantize:  # pylint: disable=too-many-instance-attributes
    """Configuration for group quantization"""

    name: str
    kind: str
    group_size: int
    quantize_dtype: Literal["int3", "int4", "int8"]
    storage_dtype: Literal["uint32"]
    model_dtype: Literal["float16", "float32"]
    linear_weight_layout: Literal["KN", "NK"]
    quantize_embedding: bool = True
    quantize_final_fc: bool = True

    num_elem_per_storage: int = 0
    num_storage_per_group: int = 0
    max_int_value: int = 0
    tensor_parallel_shards: int = 0

    def __post_init__(self):
        assert self.kind == "group-quant"
        quantize_dtype = DataType(self.quantize_dtype)
        storage_dtype = DataType(self.storage_dtype)
        model_dtype = DataType(self.model_dtype)
        assert quantize_dtype.type_code == DataTypeCode.INT
        assert storage_dtype.type_code == DataTypeCode.UINT
        assert model_dtype.type_code == DataTypeCode.FLOAT
        if storage_dtype.bits < quantize_dtype.bits:
            raise ValueError("Storage unit should be greater or equal to quantized element")

        self.num_elem_per_storage = storage_dtype.bits // quantize_dtype.bits # 都是使用uint32来保存的，则会存8个int4
        if self.group_size % self.num_elem_per_storage != 0:  # 如int4量化，分组大小应是8的倍数，否则需要一个uint32里可能会跨两个组，使量化变得更复杂
            raise ValueError("Group size should be divisible by numbers of elements per storage")
        self.num_storage_per_group = self.group_size // self.num_elem_per_storage # 一个组有多少个uint32
        self.max_int_value = (2 ** (quantize_dtype.bits - 1)) - 1 # 极大值，如在int4时，为2**3-1=7，即0111
        # linear_weight_layout分NK和KN两个内存布局，N表示batch_size或序列长度等，K表示特征维度或神经元数量等。
        # NK是一个N内的K连续，KN是一个K内的N连续。
        # mlc_llm convert_weight命令中输入的q4f16_1 采用的是NK，即使每个batch下的K个特征数据被连续访问到。
        # 而q4f16_0采用的使KN，q4f16_0和q4f16_1唯一的区别就是权重的布局KN/NK不一样。
        # CJM_TODO: linear_quant_axis的用途
        self.linear_quant_axis = 0 if self.linear_weight_layout == "KN" else 1
        self._quantize_func_cache = {}

    def quantize_model(
        self,
        model: nn.Module,
        quant_map: QuantizeMapping,
        name_prefix: str,
    ) -> nn.Module:
        """
        Quantize model with group quantization

        Parameters
        ----------
        model : nn.Module
            The non-quantized nn.Module.

        quant_map : QuantizeMapping
            The quantize mapping with name mapping and func mapping.

        name_prefix : str
            The name prefix for visited weight.

        Returns
        -------
        ret : nn.Module
            The quantized nn.Module.
        """

        # nn.Mutator里提供了基础的visit函数，可以递归遍历所有模块。
        # 这里基于nn.Mutator派生了一个子类，并重写了visit_module函数，然后通过visit函数作为入口进行遍历。
        # 每到达一个nn.Module时，会执行该重写的visit_module，从而使所有nn.Module都能执行到visit_module函数内的变换。
        class _Mutator(nn.Mutator):
            def __init__(self, config: GroupQuantize, quant_map: QuantizeMapping) -> None:
                super().__init__()
                self.config = config
                self.quant_map = quant_map

            def visit_module(self, name: str, node: nn.Module) -> Any:
                """
                The visiting method for group quantization of nn.Module nodes.

                Parameters
                ----------
                name : str
                    The name of the current node.

                node : nn.Module
                    The current node of nn.Module to mutate.

                Returns
                ------
                ret_node: Any
                    The new node to replace current node.
                """
                if (
                    isinstance(node, nn.Linear)
                    and (not is_final_fc(name) or self.config.quantize_final_fc)
                    and not is_moe_gate(name, node)
                ):
                    weight_name = f"{name}.weight"
                    # param_map 以weight名字为索引，存放对应的量化后的weight名字及其scale
                    self.quant_map.param_map[weight_name] = [f"{name}.q_weight", f"{name}.q_scale"]
                    # map_func 以weight名字为索引，存放权重量化函数quantize_weight，
                    # 并将 quantize_weight 的output_transpose参数固定住，如果layout是KN=需要转置，NK=不需要转置。
                    # quantize_weight 函数在加载权重时会调用，完成量化后再加载(loader\huggingface_loader.py#121)。
                    # 由此可见，三种可量化的层均采用quantize_weight函数为其量化，仅有 output_transpose 的差别。
                    self.quant_map.map_func[weight_name] = partial(
                        self.config.quantize_weight,
                        output_transpose=self.config.linear_weight_layout == "KN",
                    )
                    # 基于非量化的nn.linear节点(本次visit_module输入的node)去构建GroupQuantizeLinear对象作为新node返回。
                    return GroupQuantizeLinear.from_linear(node, self.config)
                if isinstance(node, nn.Embedding) and self.config.quantize_embedding:
                    weight_name = f"{name}.weight"
                    self.quant_map.param_map[weight_name] = [f"{name}.q_weight", f"{name}.q_scale"]
                    # quantize_weight未固化参数，其output_transpose参数默认为False
                    self.quant_map.map_func[weight_name] = self.config.quantize_weight
                    return GroupQuantizeEmbedding.from_embedding(node, self.config)
                if isinstance(node, MixtralExperts):
                    weight_name = f"{name}.weight"
                    self.quant_map.param_map[weight_name] = [f"{name}.q_weight", f"{name}.q_scale"]
                    self.quant_map.map_func[weight_name] = self.config.quantize_weight
                    return GroupQuantizeMixtralExperts.from_mixtral_experts(node, self.config)
                # 针对上面三种层，构建新的量化节点，分别是GroupQuantizeLinear / GroupQuantizeEmbedding / GroupQuantizeMixtralExperts，
                # 如果不是这三种情况，则继续往下遍历。
                # 完成遍历后，所有node中，包含上面三种情况的node都将会被转换为对应的量化node，并注册有quantize_weight函数。
                return self.visit(name, node)

        # 递归地将nn.Module转换为指定类型，model_dtype是"float16"或"float32"，TODO:原因？ 
        model.to(dtype=self.model_dtype)
        mutator = _Mutator(self, quant_map)
        # 遍历并执行相关变换
        model = mutator.visit(name_prefix, model)
        return model

    def _dequantize(
        self,
        weight: te.Tensor,
        scale: te.Tensor,
        axis: int,
        out_shape: Optional[List[tir.PrimExpr]] = None,
    ):
        tir_max_int = tir.const(self.max_int_value, self.model_dtype)
        # 将打包在uint32中的量化数据，逐个拆分出来用浮点存储
        float_weight = convert_uint_to_float(
            weight,
            DataType(self.quantize_dtype).bits,
            self.num_elem_per_storage,
            self.storage_dtype,
            self.model_dtype,
            axis=axis,
            out_shape=out_shape,
        )
        if out_shape is None:
            out_shape = weight.shape
            out_shape[axis] *= self.num_elem_per_storage
        axis = axis if axis >= 0 else len(out_shape) + axis
        # 减去max_int_value后乘以scale，完成反量化
        # int4量化中，max_int_value是7，在量化时做了加max_int_value的操作
        return te.compute(
            shape=out_shape,
            fcompute=lambda *idx: tir.multiply(
                tir.subtract(
                    float_weight(*idx),
                    tir_max_int,
                ),
                scale(*idx[:axis], idx[axis] // self.group_size, *idx[axis + 1 :]),
            ),
            name="dequantize",
        )

    # 实际量化操作的函数，在quantize_model函数中被注册到量化表里，在权重加载的loader函数中被调用，完成权重的量化。
    # output_transpose参数对于nn.Linear来说, 权重layout为"KN"时为True, 其他情况都为False. 
    # axis: 在权重加载时(loader\huggingface_loader.py#155)获得, 如未获得, 则默认为是最后一维.
    def quantize_weight(
        self, weight: NDArray, axis: int = -1, output_transpose: bool = False
    ) -> List[NDArray]:
        """
        Quantize weight with group quantization

        Parameters
        ----------
        weight : NDArray
            The original weight.

        axis : int
            The group axis.

        output_transpose : bool
            Whether to transpose the output quantized weight. Only 2D weight is supported.

        Returns
        ------
        ret: List[NDArray]
            The list of group quantized weights.
        """
        device = weight.device
        device_type = device.MASK2STR[device.device_type]
        axis = axis if axis >= 0 else len(weight.shape) + axis  # axis小于0时,为默认的-1, 即此时asix是权重的最后一维.

        # TVM基础 dataflow上下文 为代码提供了明确的数据流向标识、逻辑隔离以及优化和并行性的暗示，有助于提高代码的可读性、可维护性和执行效率
        # 1. 可以明确地标识了一个区域，在这个区域内的操作主要关注数据的流动和转换。
        #    下面将与量化相关操作放在dataflow块内，可以清晰地看出从输入的weight_var到最终输出的整个数据处理流程是围绕量化这个核心任务展开的。
        # 2. 隔离数据处理逻辑。
        
        # 创建量化函数，类型是tvm.IRModule
        # 使用 Relax 模块，将weight的shape和type作为类型信息，构建一个 weight_var。
        # 使用bb.function创建一个名为main的函数，输入参数是 weight_var。
        # 函数内，首先进入数据流式上下文，将weight_var和一些其他信息作为绑定参数，提供给_quantize函数，作为一个张量表达式操作TE, 
        # 使用bb.emit_te发射出，将结果lv赋值给gv，作为函数返回值。使用bb.finalize()结束IRModule的构建。
        def _create_quantize_func() -> IRModule:
            bb = relax.BlockBuilder()  # pylint: disable=invalid-name
            weight_var = relax.Var("weight", relax.TensorStructInfo(weight.shape, weight.dtype))
            with bb.function(name="main", params=[weight_var]):
                with bb.dataflow():
                    lv = bb.emit_te(self._quantize, weight_var, axis, output_transpose)
                    gv = bb.emit_output(lv)  # pylint: disable=invalid-name
                bb.emit_func_output(gv)
            return bb.finalize()

        # 针对围绕key编译量化函数，key包含的参数如下，即每个量化函数均值针对指定的shape/dtype/device/asix/output_transpose来编译的。
        # 如不同的量化层，其对应的key上的参数是一样的，则直接使用之前编译过的量化函数即可。
        key = (
            f"({weight.shape}, {weight.dtype}, {device_type}, "
            f"axis={axis}, output_transpose={output_transpose})"
        )
        quantize_func = self._quantize_func_cache.get(key, None)
        if quantize_func is None:
            logger.info("Compiling quantize function for key: %s", key)
            quantize_func = compile_quantize_func(_create_quantize_func(), device=device)
            self._quantize_func_cache[key] = quantize_func
        # 编译或选取量化函数，执行量化
        return quantize_func(weight)

    # 执行量化的最内层的TE函数。
    def _quantize(  # pylint: disable=too-many-locals
        self,
        weight: te.Tensor,
        axis: int = -1,
        output_transpose: bool = False,
    ) -> Tuple[te.Tensor, te.Tensor]:
        """Group quantization for weight tensor, defined in tensor expression."""
        # max_int_value对于int4量化来说是7，即0111; model_dtype是"float16"或"float32".
        max_int = tir.const(self.max_int_value, self.model_dtype)
        shape = weight.shape  # pylint: disable=invalid-name
        axis = axis if axis >= 0 else len(shape) + axis
        k = shape[axis]
        #######
        # compute scale per group
        # 定义一个规约轴，范围是从0到group_size. 即 r 将遍历 [0, 1, 2, ..., group_size - 1]。
        r = te.reduce_axis((0, self.group_size), name="r")  # pylint: disable=invalid-name
        # ceildiv是向上取整，维度k除以group_size 得到group数量.
        num_group = tir.ceildiv(k, self.group_size)
        # 将原始形状shape在特定维度axis处进行分割，插入计算得到的num_group，从而得到新的形状.
        # shape[:axis]是从0到axis, 不包含axis; shape[axis + 1 :]也不包含axis.
        # 如 shape = (4, 8, 16), axis = 1, num_group = 2, 则得到 (4, 2, 16).
        #    shape = (3584, 2368), axis = 1, num_group = 592, 则得到 (3584, 592).
        scale_shape = (*shape[:axis], num_group, *shape[axis + 1 :])
        
        # 创建一个名为max_abs的张量计算（TE compute）操作, 该张量的shape为刚得到的scale_shape, 即会基于scale_shape进行循环遍历.
        # 使用了循环变量idx, 这里取*号,是一个元组, 如shape为(4, 8),那么idx是(i, j)，其中i遍历[0, 1, 2, 3]，j遍历[0, 1, 2, 3, 4, 5, 6, 7]。
        # fcompute指定为一个匿名函数, 输入是循环变量idx元组(循环已被省略掉,不用写出来). 
        # 匿名函数最内层tir.if_then_else内进行条件判断, 判断条件是 idx[axis] * self.group_size + r < k, 
        # 如shape为(3584, 2368),而axis是最后一维, 则idx[axis]将会是for循环遍历j维度上的0-2367, j 维度上一个数值对应一个group, 
        #   乘以group_size则跳到权重里某个分组上; r 是上面定义的规约轴, 会遍历从0到group_size-1, 所以会再套一层r的循环.
        #   一层遍历 idx[axis]的j维度上的0-2367, 一层遍历r的0到group_size-1. 
        # idx[axis] * self.group_size + r < k, 表示所选取的位置在权重的范围内, 则取该位置上的值的绝对值; 
        #   如果超出范围,如k维度只有10,分组大小是4,分三组,则最后一组只有一半的数据, 则取float16或float32的最小值.
        # te.abs是先切片, 前后分片*idx[:axis]和*idx[axis + 1 :],留下*idx[axis]这个维度上的数据, 
        #   取其idx[axis] * self.group_size + r位置上的数据,计算绝对值.
        # tir.if_then_else这一层的绝对值取完后, 再由te.max, 基于r维度, 寻找最大值, 并返回.
        # 
        # 简言之: 如对于weight_shape[4,10], 以axis=-1进行分组,group_size为4,分3组,即分别是(0-3)(4-7)(8-9), scale_shape会是[4,3]
        # 内层tir.if_then_else的第一层循环会遍历scale_shape的-1维度上三个组,即0,1,2, 第二层循环会遍历围绕group_size的r的0,1,2,3.
        # 则对于weight中[4,10]每个元素均取其绝对值, 因为按idx[axis] * self.group_size + r < k会取到[4,12]的范围, 则超出[4,10]的部分取值为te.min_value(self.model_dtype).
        # 取完绝对值后, te.max再基于规约轴r, 即在[4,0-3]/[4,4-7]/[4,8-9]的各自范围内计算最大值, 得到[4,3]个max_abs.
        max_abs = te.compute(
            shape=scale_shape,
            fcompute=lambda *idx: te.max(
                tir.if_then_else(
                    idx[axis] * self.group_size + r < k,
                    te.abs(weight(*idx[:axis], idx[axis] * self.group_size + r, *idx[axis + 1 :])),
                    te.min_value(self.model_dtype),
                ),
                axis=r,
            ),
            name="max_abs_value",
        )
        # 创建一个名为max_abs的张量计算,该张量的shape为同样是scale_shape.
        # 将分组内最大的绝对值除以max_int,得到scale. 基于上面例子,维度同样是[4,3]
        scale = te.compute(
            scale_shape,
            lambda *idx: max_abs(*idx).astype(self.model_dtype) / max_int,
            name="scale",
        )
        ######
        # compute scaled weight
        # 基于weight_shape遍历, 循环变量idx仍使用*修饰, 即将索引参数解包传入,是个多维元组.
        # 对于weight的每一个元素 weight(*idx), 
        # 除以其对应的scale值 (基于axis维度,前后分片,只留下axis部分显示处理,前后分片处照常展开处理即可; 
        #                     取weight权重的axis维度上被group_size整除的对应下标的值)
        # 再加上max_int (int4中对应7) 得到 量化后的参数. 
        # 并将数值范围限制在 [0, 2*max_int] 之内。
        # .astype(self.storage_dtype), 最后将类型转为uint32进行保存。
        #
        # CJM_TODO: int4量化时，为什么要+7?
        scaled_weight = te.compute(
            shape=weight.shape,
            fcompute=lambda *idx: tir.min(
                tir.max(
                    tir.round(
                        weight(*idx)
                        / scale(*idx[:axis], idx[axis] // self.group_size, *idx[axis + 1 :])
                        + max_int
                    ),
                    tir.const(0, self.model_dtype),
                ),
                max_int * 2,
            ).astype(self.storage_dtype),
        )
        ######
        # compute quantized weight per storage
        # 因为一个元素不足一个字节，所以需要进行打包，以storage_dtype为单元进行存储。
        num_storage = self.num_storage_per_group * num_group
        quantized_weight_shape = (*shape[:axis], num_storage, *shape[axis + 1 :])
        quantized_weight = pack_weight(
            scaled_weight,
            axis=axis,
            num_elem_per_storage=self.num_elem_per_storage,
            weight_dtype=self.quantize_dtype,
            storage_dtype=self.storage_dtype,
            out_shape=quantized_weight_shape,
        )
        # 如需要转置，直接调用topi算子库进行转置操作。
        if output_transpose:
            if len(quantized_weight.shape) != 2 or len(scale.shape) != 2:
                raise ValueError(
                    "Does not support transpose output quantized weight with ndim != 2"
                )
            quantized_weight = topi.transpose(quantized_weight)
            scale = topi.transpose(scale)
        return quantized_weight, scale


class GroupQuantizeLinear(nn.Module):  # pylint: disable=too-many-instance-attributes
    """An nn.Linear module with group quantization"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_features: int,
        out_features: Union[int, tir.Var],
        config: GroupQuantize,
        bias: bool = True,
        out_dtype: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_dtype = out_dtype
        self.config = config
        num_group = tir.ceildiv(in_features, config.group_size)
        num_shards = config.tensor_parallel_shards
        if num_shards > 1 and (in_features * num_shards // config.group_size) % num_shards != 0:
            raise ValueError(
                f"The linear dimension {in_features * num_shards} has "
                f"{in_features * num_shards // config.group_size} groups under group size "
                f"{config.group_size}. The groups cannot be evenly distributed on "
                f"{num_shards} GPUs.\n"
                "Possible solutions: reduce number of GPUs, or use quantization with smaller "
                "group size."
            )
        if config.linear_weight_layout == "KN":
            self.q_weight = nn.Parameter(
                (config.num_storage_per_group * num_group, out_features), config.storage_dtype
            )
            self.q_scale = nn.Parameter((num_group, out_features), config.model_dtype)
        else:
            self.q_weight = nn.Parameter(
                (out_features, config.num_storage_per_group * num_group), config.storage_dtype
            )
            self.q_scale = nn.Parameter((out_features, num_group), config.model_dtype)
        if bias:
            self.bias = nn.Parameter(
                (out_features,), config.model_dtype if out_dtype is None else out_dtype
            )
        else:
            self.bias = None

    @staticmethod
    def from_linear(src: nn.Linear, config: GroupQuantize) -> "GroupQuantizeLinear":
        """
        Converts a non-quantized nn.Linear to a group quantized GroupQuantizeLinear

        Parameters
        ----------
        src : nn.Linear
            The non-quantized nn.Linear.

        config : GroupQuantize
            The group quantization config.

        Returns
        -------
        ret : GroupQuantizeLinear
            The group quantized GroupQuantizeLinear layer.
        """
        # For dynamic shape, src.out_features is `"name"`; src.weight.shape[0] is `tir.Var("name")`
        out_features, in_features = src.weight.shape
        quantized_linear = GroupQuantizeLinear(
            in_features=in_features,
            out_features=out_features,
            config=config,
            bias=getattr(src, "bias", None) is not None,
            out_dtype=src.out_dtype,
        )
        if quantized_linear.bias is not None:
            quantized_linear.bias.attrs = src.bias.attrs
        if "shard_strategy" in src.weight.attrs:
            shard = src.weight.attrs["shard_strategy"]
            apply_sharding(shard, f"{shard.name}_q_weight", quantized_linear.q_weight)
            apply_sharding(shard, f"{shard.name}_q_scale", quantized_linear.q_scale)
        return quantized_linear

    def forward(self, x: nn.Tensor) -> nn.Tensor:  # pylint: disable=invalid-name
        """
        Forward method for group quantized linear layer.

        Parameters
        ----------
        x : nn.Tensor
            The input tensor.

        Returns
        -------
        ret : nn.Tensor
            The output tensor for the group quantized linear layer.
        """
        w = nn.op.tensor_expr_op(  # pylint: disable=invalid-name
            lambda weight, scale: self.config._dequantize(  # pylint: disable=protected-access
                weight,
                scale,
                axis=self.config.linear_quant_axis,
                out_shape=(
                    [
                        (
                            tir.IntImm("int64", self.out_features)
                            if isinstance(self.out_features, int)
                            else weight.shape[0]
                        ),  # Reuse same tir.Var for symbolic shape (after Exporter)
                        tir.IntImm("int64", self.in_features),
                    ]
                    if self.config.linear_weight_layout == "NK"
                    else [
                        tir.IntImm("int64", self.in_features),
                        (
                            tir.IntImm("int64", self.out_features)
                            if isinstance(self.out_features, int)
                            else weight.shape[1]
                        ),  # Reuse same tir.Var for symbolic shape (after Exporter)
                    ]
                ),
            ),
            name_hint="dequantize",
            args=[self.q_weight, self.q_scale],
        )
        if self.config.linear_weight_layout == "NK":
            w = nn.op.permute_dims(w)  # pylint: disable=invalid-name
        x = nn.op.matmul(x, w, out_dtype=self.out_dtype)
        if self.bias is not None:
            x = x + self.bias
        return x

    def to(self, dtype: Optional[str] = None) -> None:
        """
        Override to() such that we do not convert bias if there is an out_dtype.
        Otherwise, we might run into dtype mismatch when computing x + self.bias.
        """
        self.q_weight.to(dtype=dtype)
        self.q_scale.to(dtype=dtype)
        if self.bias is not None and self.out_dtype is None:
            self.bias.to(dtype=dtype)
        if dtype is not None and isinstance(getattr(self, "dtype", None), str):
            self.dtype = dtype  # pylint: disable=attribute-defined-outside-init


class GroupQuantizeEmbedding(nn.Module):
    """An nn.Embedding module with group quantization"""

    def __init__(self, num: Union[int, tir.Var], dim: int, config: GroupQuantize):
        self.num = num
        self.dim = dim
        self.config = config
        num_group = tir.ceildiv(dim, config.group_size)
        self.q_weight = nn.Parameter(
            (num, config.num_storage_per_group * num_group), config.storage_dtype
        )
        self.q_scale = nn.Parameter((num, num_group), config.model_dtype)

    @staticmethod
    def from_embedding(embedding: nn.Embedding, config: GroupQuantize) -> "GroupQuantizeEmbedding":
        """
        Converts a non-quantized nn.Embedding to a group quantized GroupQuantizeEmbedding

        Parameters
        ----------
        linear : nn.Embedding
            The non-quantized nn.Embedding.

        config : GroupQuantize
            The group quantization config.

        Returns
        -------
        ret : GroupQuantizeEmbedding
            The group quantized GroupQuantizeEmbedding layer.
        """
        num, dim = embedding.weight.shape
        return GroupQuantizeEmbedding(num, dim, config)

    def forward(self, x: nn.Tensor):  # pylint: disable=invalid-name
        """
        Forward method for group quantized embedding layer.

        Parameters
        ----------
        x : nn.Tensor
            The input tensor.

        Returns
        -------
        ret : nn.Tensor
            The output tensor for the embedding layer.
        """
        w = nn.op.tensor_expr_op(  # pylint: disable=invalid-name
            lambda weight, scale: self.config._dequantize(  # pylint: disable=protected-access
                weight,
                scale,
                axis=-1,
                out_shape=[
                    (
                        tir.IntImm("int64", self.num)
                        if isinstance(self.num, int)
                        else weight.shape[0]
                    ),  # Reuse same tir.Var for symbolic shape (after Exporter)
                    tir.IntImm("int64", self.dim),
                ],
            ),
            name_hint="dequantize",
            args=[self.q_weight, self.q_scale],
        )
        if x.ndim == 1:
            return nn.op.take(w, x, axis=0)
        return nn.op.reshape(
            nn.op.take(w, nn.op.reshape(x, shape=[-1]), axis=0),
            shape=[*x.shape, self.dim],
        )

    def lm_head_forward(self, x: nn.Tensor):
        """The lm_head forwarding, which dequantizes the weight
        and multiplies it with the input tensor.

        Parameters
        ----------
        x : nn.Tensor
            The input tensor.

        Returns
        -------
        ret : nn.Tensor
            The output tensor for the lm_head layer.
        """
        w = nn.op.tensor_expr_op(  # pylint: disable=invalid-name
            lambda weight, scale: self.config._dequantize(  # pylint: disable=protected-access
                weight,
                scale,
                axis=-1,
                out_shape=[
                    (
                        tir.IntImm("int64", self.num)
                        if isinstance(self.num, int)
                        else weight.shape[0]
                    ),
                    tir.IntImm("int64", self.dim),
                ],
            ),
            name_hint="dequantize",
            args=[self.q_weight, self.q_scale],
        )
        w = nn.op.permute_dims(w)
        return nn.op.matmul(x, w, out_dtype="float32")


class GroupQuantizeMixtralExperts(nn.Module):  # pylint: disable=too-many-instance-attributes
    """An MixtralExperts module with group quantization"""

    def __init__(
        self,
        num_local_experts,
        in_features,
        out_features,
        config: GroupQuantize,
    ):  # pylint: disable=too-many-arguments
        self.num_local_experts = num_local_experts
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        num_group = tir.ceildiv(in_features, config.group_size)
        self.q_weight = nn.Parameter(
            (num_local_experts, out_features, config.num_storage_per_group * num_group),
            config.storage_dtype,
        )
        self.q_scale = nn.Parameter(
            (num_local_experts, out_features, num_group), config.model_dtype
        )
        self.quantize_dtype = config.quantize_dtype
        self.group_size = config.group_size
        self.dtype = config.model_dtype
        if config.linear_weight_layout == "KN":
            raise NotImplementedError("GroupQuantizeMixtralExperts does not support KN layout now.")

    @staticmethod
    def from_mixtral_experts(
        src: "MixtralExperts", config: GroupQuantize
    ) -> "GroupQuantizeMixtralExperts":
        """
        Converts a non-quantized MixtralExperts to a group quantized GroupQuantizeMixtralExperts

        Parameters
        ----------
        src : MixtralExperts
            The non-quantized MixtralExperts

        config : GroupQuantize
            The group quantization config.

        Returns
        -------
        ret : GroupQuantizeMixtralExperts
            The group quantized GroupQuantizeMixtralExperts layer.
        """
        quantized_mistral_experts = GroupQuantizeMixtralExperts(
            num_local_experts=src.num_local_experts,
            in_features=src.in_features,
            out_features=src.out_features,
            config=config,
        )
        if "shard_strategy" in src.weight.attrs:
            shard = src.weight.attrs["shard_strategy"]
            apply_sharding(shard, f"{shard.name}_q_weight", quantized_mistral_experts.q_weight)
            apply_sharding(shard, f"{shard.name}_q_scale", quantized_mistral_experts.q_scale)
        return quantized_mistral_experts

    def forward(self, x: nn.Tensor, indptr: nn.Tensor) -> nn.Tensor:  # pylint: disable=invalid-name
        """Forward method for group quantized mistral experts.

        Parameters
        ----------
        x : nn.Tensor
            The input tensor.

        indptr: nn.Tensor
            The indptr tensor

        single_batch_decode: bool
            Whether to use single-batch decode

        Returns
        -------
        ret : nn.Tensor
            The output tensor for the group quantized mistral experts layer.
        """
        from mlc_llm.op import moe_matmul  # pylint: disable=import-outside-toplevel

        assert x.ndim == 2
        if indptr.ndim == 2:  # single-batch
            assert indptr.shape[0] == 1
            return moe_matmul.dequantize_gemv(
                x,
                self.q_weight,
                self.q_scale,
                indptr,
                quantize_dtype=self.quantize_dtype,
                group_size=self.group_size,
            )
        assert indptr.ndim == 1
        return moe_matmul.dequantize_group_gemm(
            x,
            self.q_weight,
            self.q_scale,
            indptr,
            quantize_dtype=self.quantize_dtype,
            indptr_dtype=indptr.dtype,
            group_size=self.group_size,
        )
