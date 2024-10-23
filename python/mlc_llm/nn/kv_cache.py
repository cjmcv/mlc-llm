"""Attention KV cache modeling."""

# pylint: disable=too-many-statements,too-many-lines,too-many-arguments
import json
from typing import Any, Dict, List, Optional

import numpy as np
from tvm import relax as rx
from tvm import tir
from tvm.relax.frontend.nn.llm.kv_cache import PagedKVCache as TVMPagedKVCache
from tvm.relax.frontend.nn.llm.kv_cache import RopeMode

## python基础
# 函数返回"PagedKVCache"，加了双引号，表示这是前向引用。
# 即在类PagedKVCache内部定义了一个create_generic方法，这个方法返回的是这个PagedKVCache类本身，
# 但是在定义这个方法的时候，类PagedKVCache可能还没有被完全定义。所以加双引号告诉python解析器，"PagedKVCache"是一个稍后会被定义的类型名称，而不是一个未定义的变量。

# 这个类继承于tvm里的PagedKVCache (tvm\python\tvm\relax\frontend\nn\llm\kv_cache.py#83), 
# 同一文件下有FlashInferPagedKVCache 和 TIRPagedKVCache, 均和该文件下的PagedKVCache为同一级别类，均为TVMPagedKVCache的派生类。
# 这个类仅有一个create_generic的静态方法，通过rx.call_pure_packed(...)创建表达式_expr, 和name "paged_kv_cache" 一起作为构造函数的参数，构造PagedKVCache类。
#
# 回溯到最顶层的父类是Object(tvm\python\tvm\relax\frontend\nn\core.py#299), 
# Object是relax顶层的一个wrapper，它可以高效地在编译pipeline中，完成对子类的替换。
# Object构造函数提供的参数之一的Expr的struct_info，是基础的Object结构信息，而不是其子类的。
# 而这里的_expr=rx.call_pure_packed(...)打包的就是基础的Object结构信息，而不针对其子类。
# 子类有FlashInferPagedKVCache, TIRPagedKVCache 以及 当前文件下的 PagedKVCache，共三个。
# 结合DispatchKVCacheCreation (mlc_llm\compiler_pass\dispatch_kv_cache_creation.py#56)
# 在编译pipeline中，经过调优会选择其中一个进行替换。
#
# rx.call_pure_packed的主体"mlc.create_paged_kv_cache_generic" 
# 与 extract_creation_args 相对应 (compiler_pass\dispatch_kv_cache_creation.py#22)，作为 global_symbol 存在。
# 在 DispatchKVCacheCreation (mlc_llm\compiler_pass\dispatch_kv_cache_creation.py#)的 transform_module 
# 会调用 create_tir_paged_kv_cache 和 create_flashinfer_paged_kv_cache，并传入参数（包含这个global_symbol）
#
# CJM_TODO：把tir和flashinfer的paged_kv_cache实现都加入编译优化的搜索范围内，通过这个global_symbol来绑定，使二者放在一起比较和选择？？
#
# 以TIRPagedKVCache 为例，它的构造是将很多tvmscript写的优化函数作为参数打包进 vm.builtin.paged_attention_kv_cache_create_reduced 中。
# 会进而送到(3rdparty\tvm\src\runtime\relax_vm\paged_kv_cache.cc#2664)的各个PackedFunc中，由c++端使用这些函数指针。
# 推理计算时的调用链路为：QWen2Attention.forward (model\qwen2\qwen2_model.py#145)
#                         -> paged_kv_cache.attention_with_fused_qkv
#                            -> vm.builtin.attention_kv_cache_attention_with_fused_qkv 
#                               (tvm\relax\frontend\nn\llm\kv_cache.py#111)
#                               (tvm\src\runtime\relax_vm\kv_state.cc#73)
#                               -> AttentionWithFusedQKV (tvm\src\runtime\relax_vm\paged_kv_cache.cc#1729)
#                                  -> 基于上面从python端打包进入vm.builtin.paged_attention_kv_cache_create_reduced的tvmscript优化算子完成kvcache的计算。
class PagedKVCache(TVMPagedKVCache):  # pylint: disable=too-few-public-methods
    """The Paged KV Cache used in LLM batching for efficient attention computation."""

    @staticmethod
    def create_generic(  # pylint: disable=too-many-locals
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        prefill_chunk_size: tir.Var,
        page_size: tir.Var,
        support_sliding_window: tir.Var,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rope_mode: RopeMode,
        rope_scale: int,
        rope_theta: int,
        dtype: str,
        rotary_dim: Optional[int] = None,
        rope_scaling: Optional[Dict[str, Any]] = None,
        rope_ext_factors: Optional[List[int]] = None,
        layer_partition: Optional[List[int]] = None,
        name: str = "paged_kv_cache",
    ) -> "PagedKVCache":
        """The generic function of creating a PagedKVCache,
        which will be rewritten by functions in compilation pipeline.
        """
        if rotary_dim is None:
            rotary_dim = head_dim
        if rope_scaling is None:
            rope_scaling = {}
        if layer_partition is None:
            layer_partition = [0, num_hidden_layers]
        return PagedKVCache(
            _expr=rx.call_pure_packed(
                "mlc.create_paged_kv_cache_generic",
                rx.ShapeExpr(
                    [
                        max_batch_size,
                        max_total_seq_len,
                        prefill_chunk_size,
                        page_size,
                        support_sliding_window,
                    ]
                ),
                rx.ShapeExpr(layer_partition),
                rx.PrimValue(num_hidden_layers),
                rx.PrimValue(num_attention_heads),
                rx.PrimValue(num_key_value_heads),
                rx.PrimValue(head_dim),
                rx.PrimValue(rope_mode),
                rx.PrimValue(rope_scale),
                rx.PrimValue(rope_theta),
                rx.StringImm(json.dumps(rope_scaling)),
                (
                    rx.const(np.array(rope_ext_factors, "float32"))
                    if rope_ext_factors is not None
                    else rx.PrimValue(0)
                    # NOTE: since relax does not have "Optional" type, we use PrimValue(0)
                    # to represent "undefined".
                ),
                rx.PrimValue(rotary_dim),
                rx.DataTypeImm(dtype),
                sinfo_args=rx.ObjectStructInfo(),
            ),
            _name=name,
        )
