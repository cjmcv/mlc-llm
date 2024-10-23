"""
Implementation for QWEN2 architecture.
"""

import dataclasses
from functools import partial
from typing import Any, Dict, Optional

from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.support import logging
from mlc_llm.support import tensor_parallel as tp
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

logger = logging.getLogger(__name__)


# tie_word_embeddings 是一种优化技术，用于将模型的词嵌入层与其输出层的权重矩阵进行绑定（tie），即让它们共享相同的权重矩阵。这意味着输入层的词向量和输出层的概率分布是通过同一个权重矩阵进行转换的。
#                     这种技术可以减少模型的参数数量，提高训练效率。在某些情况下，绑定权重矩阵可以提高模型的性能，因为它强制模型在输入和输出之间保持一致的语义表示。
# * 词嵌入层：如Qwen2Embedding，将输入文本中的每个词（或子词）映射到一个固定维度的向量空间中的层。这些向量捕捉了词的语义信息，使得语义相似的词在向量空间中距离较近。
# * 输出层权重矩阵：在语言模型中，输出层通常是一个全连接层，其权重矩阵用于将隐藏层的输出映射到词汇表中的每个词的概率分布。
# 不使用tie_word_embeddings的情况：如在多任务学习场景中，不同的任务可能需要不同的词嵌入表示，绑定权重矩阵可能会限制模型的灵活性。
#                                 如在特定领域的任务中，某些特定领域的任务可能需要更复杂或更专门的词嵌入表示，绑定权重矩阵可能无法满足这些需求。
#                                 语义差异：词嵌入层和输出层可能需要捕捉不同的语义信息，绑定权重矩阵可能会限制这种灵活性。
#                                 过拟合风险：在某些数据集上，绑定权重矩阵可能会增加过拟合的风险，特别是在数据量较小的情况下。
#
# tensor_parallel_shards 张量并行的分片数量
# * 张量并行是将模型层内的参数（张量）切分到不同的计算设备上，实现层内并行。
# * 流水线并行是将模型的不同层（如Transformer层）分配到不同的计算设备上，形成一条计算流水线,每个设备负责模型的一部分层。
# 层内参数太大，用张量并行；层内参数小，层数多，用流水线并行。
# 对于一次推理过程，张量并行可以起到加速作用，而流水线并行则因为模型层的顺序性而无法直接加速推理过程。
#
# num_attention_heads和num_key_value_heads的关系
# * 在标准的多头注意力实现中，num_key_value_heads会等于num_attention_heads，即每个注意力头都有自己独立的 Q、K、V 计算。
#   某些情况下，设置num_key_value_heads小于num_attention_heads是为了节省计算资源，当计算 K 和 V 时需要相对较多的计算资源（例如在处理长序列数据时，K 和 V 的矩阵乘法计算量较大），通过减少用于计算 K 和 V 的头的数量，可以在一定程度上降低计算成本
#              也可看成是信息融合的方式，较少数量的 K 和 V 头可以对信息进行一种粗粒度的整合，然后由更多数量的注意力头基于这些整合后的 K 和 V 来进行更细致的注意力计算。
@dataclasses.dataclass
class QWen2Config(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the QWen2 model."""
    hidden_act: str    # 隐藏层的激活函数类型
    hidden_size: int   # 模型隐藏层的维度大小
    intermediate_size: int    # 中间层的大小
    num_attention_heads: int  # 注意力头的数量
    num_hidden_layers: int    # 隐藏层的数量
    num_key_value_heads: int  # 键值对注意力头的数量
    rms_norm_eps: float       # RMS(均方根)归一化的epsilon值，用于防止分母为零，起到稳定训练的作用; rms = math.sqrt((sum(x_i**2 for x_i in data) + epsilon) / (len(data) + epsilon))
    rope_theta: int           # RoPE（旋转位置编码）的参数，影响位置编码的方式? CJM_TODO: 具体作用，用在哪里?
    vocab_size: int           # 词汇表的大小，即模型能够处理的不同单词或标记的数量
    tie_word_embeddings: bool = False  # 如果为 True，表示将输入的词嵌入与输出的词嵌入绑定，即共享相同的权重矩阵。
    context_window_size: int = 0       # 上下文窗口的大小，决定了模型在处理序列时能够考虑的前后文的长度. CJM_TODO: 具体作用，用在哪里?
    prefill_chunk_size: int = 0        # 预填充块的大小，与模型在处理输入数据时的分块策略有关。（在paged_kv_cache中使用）
    tensor_parallel_shards: int = 1    # 张量并行的分片数量，用于在多个设备上并行计算模型。
    head_dim: int = 0                  # 每个注意力头的维度大小
    dtype: str = "float32"
    max_batch_size: int = 1            # 最大的批处理大小（在paged_kv_cache中使用）
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.context_window_size == 0:
            for name in ["max_position_embeddings", "max_sequence_length"]:
                if name in self.kwargs:
                    self.context_window_size = self.kwargs.pop(name)
                    logger.info(
                        "%s not found in config.json. Falling back to %s (%d)",
                        bold("context_window_size"),
                        bold(name),
                        self.context_window_size,
                    )
                    break
            else:
                raise ValueError(
                    "Unable to determine the maximum sequence length, because none of "
                    "`context_window_size`, `max_position_embeddings` or `max_sequence_length` is "
                    "provided in `config.json`."
                )
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads 
        assert self.head_dim * self.num_attention_heads == self.hidden_size # head的维度 * head的数量 = 隐藏层的大小
        if self.prefill_chunk_size == 0:
            logger.info(
                "%s defaults to %d",
                bold("prefill_chunk_size"),
                min(self.context_window_size, 2048),
            )
            self.prefill_chunk_size = min(self.context_window_size, 2048)
        elif self.prefill_chunk_size > self.context_window_size:
            logger.info(
                "Overriding %s from %d to %d",
                bold("prefill_chunk_size"),
                self.prefill_chunk_size,
                min(self.context_window_size, 2048),
            )
            # context_window_size 定义了模型在一次推理过程中能够处理的最大上下文长度。
            # prefill_chunk_size要小于等于context_window_size，否则prefill时，在处理输入文本时可能会将上下文信息分割成不连续的部分，导致生成的文本缺乏连贯性。
            self.prefill_chunk_size = min(self.context_window_size, 2048)


# pylint: disable=invalid-name,missing-docstring,too-many-locals

# 一个QWen2DecoderLayer里会有一个QWen2Attention
# QWen2Attention里有 前后两个nn.Linear层 + 中间一个paged_kv_cache.attention_with_fused_qkv
# 
# paged_kv_cache:  vm.builtin.paged_attention_kv_cache_create
#   进入attention_with_fused_qkv函数（tvm\python\tvm\relax\frontend\nn\llm\kv_cache.py#86）
#   -> 调用vm.builtin.attention_kv_cache_attention_with_fused_qkv（tvm\src\runtime\relax_vm\kv_state.cc#73）
#      -> 进入到 AttentionKVCache.AttentionWithFusedQKV（tvm\src\runtime\relax_vm\paged_kv_cache.cc#1729）
#           该函数内会调用很多函数指针以完成计算，其中包含f_attention_prefill_/f_attention_decode_等
#         搜索AttentionKVCache，发现其使用了TVM_REGISTER_GLOBAL("vm.builtin.paged_attention_kv_cache_create")进行注册了创建函数，
#           这些函数指针会在该创建函数中被注册到变量中。(tvm\src\runtime\relax_vm\paged_kv_cache.cc#2586)
#         搜索注册的字段，在python端找到被使用的位置(3rdparty\tvm\python\tvm\relax\frontend\nn\llm\kv_cache.py#385), 
#           函数指针被按顺序一一使用bb.add_func进行添加。python端采用tvmscript的方式实现，由cc端调用。
class QWen2Attention(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: QWen2Config):
        self.head_dim = config.head_dim
        if config.num_key_value_heads % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split {config.num_key_value_heads} key-value attention heads "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.num_attention_heads = config.num_attention_heads // config.tensor_parallel_shards
        self.num_key_value_heads = config.num_key_value_heads // config.tensor_parallel_shards
        self.rope_theta = config.rope_theta

        # 输入维度是 hidden_size, batch_size *  seq_len？ CJM_TODO: 输入数据是什么是否要拼接？
        # 输出维度是 qkv 三个矩阵拼接在一起的维度。
        # q = self.self_attn.num_attention_heads * head_dim
        # k = self.self_attn.num_key_value_heads * head_dim
        # v = self.self_attn.num_key_value_heads * head_dim
        self.c_attn = nn.Linear(
            in_features=config.hidden_size,
            out_features=(2 * self.num_key_value_heads + self.num_attention_heads) * self.head_dim,
            bias=True,
        )
        # 输入维度是 self.num_attention_heads * self.head_dim
        # 输出维度又变回QWen2Attention输入的hidden_size的大小，即一轮QWen2Attention，维度上不产生变化。
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )
        
    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        d, h_q, h_kv = self.head_dim, self.num_attention_heads, self.num_key_value_heads
        # b*s 对应c_attn的in_features, 分别是batch_size和total_seq_len的意思。 (h_q + h_kv + h_kv*d 对应out_features。
        # prefill阶段，进的是同一个请求的s个token，则前两维是(1,s)。
        # decode阶段， 进的是b个请求的一个token，  则前两维是(b,1)。
        b, s, _ = hidden_states.shape
        qkv = self.c_attn(hidden_states)
        qkv = op.reshape(qkv, (b, s, h_q + h_kv + h_kv, d)) 
        output = op.reshape(
            paged_kv_cache.attention_with_fused_qkv(layer_id, qkv, self.num_attention_heads),
            (b, s, h_q * d),
        )
        attn_output = self.o_proj(output)
        return attn_output


ACT2FN = {
    "gelu": partial(nn.gelu, approximate=False),
    "relu": nn.relu,
    "silu": nn.silu,
    "swish": nn.silu,
    "gelu_new": partial(nn.gelu, approximate=True),
}


class Qwen2Embedding(nn.Embedding):
    """The embedding module specialized for Qwen2 so that
    it can be shared with the final lm_head.
    """

    def lm_head_forward(self, x: nn.Tensor):
        """The lm_head forwarding, which transposes the weight and multiplies
        with the input tensor.
        """
        weight = nn.op.permute_dims(self.weight)
        return nn.op.matmul(x, weight, out_dtype="float32")

# 一个QWen2DecoderLayer里会有一个QWen2MLP
# 里面包含有： 两个nn.Linear + 一个激活层
class QWen2MLP(nn.Module):
    def __init__(self, config: QWen2Config):
        if config.intermediate_size % config.tensor_parallel_shards != 0:
            raise ValueError(
                f"Cannot split MLP intermediate size {config.intermediate_size} "
                f"evenly to {config.tensor_parallel_shards} GPUs."
            )
        self.intermediate_size = config.intermediate_size // config.tensor_parallel_shards
        
        # nn.Linear即为全连接层，nn.Linear(in_features, out_features), 权重w维度为(out_features, in_features)。
        # 其输入x维度应为(batch_size, in_features), 计算y=x*w时，x的列与w的列大小一致。x的行batch_size，与输出y的行一致。
        # 实际计算过程是将权重进行转置 (out_features, in_features) -> (in_features, out_features), 充当矩阵乘法的B矩阵，进行矩阵乘法运算。
        # 简言之，全连接层是权重w转置后充当B矩阵与输入A进行矩阵乘法计算，再加上bias。
        #
        # 
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: Tensor):
        concat_x1_x2 = self.gate_up_proj(x)
        x1, x2 = op.split(concat_x1_x2, 2, axis=-1)
        return self.down_proj(self.act_fn(x1) * x2)

# 一个QWen2Model里会有num_hidden_layers个重复的一样的QWen2DecoderLayer。
# 一个QWen2DecoderLayer里有：一个QWen2Attention + 一个QWen2MLP + 两个RMSNorm。
# * 变量写的layernorm，但创建的是RMSNorm, RMSNorm与LayerNorm很相似，省略了LayerNorm部分计算，计算更高效。
#   在某些架构的语言模型中，使用 RMSNorm 代替 LayerNorm 可以在不损失模型性能的情况下，提高训练和推理的速度。
class QWen2DecoderLayer(nn.Module):
    def __init__(self, config: QWen2Config):
        self.self_attn = QWen2Attention(config)
        self.mlp = QWen2MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, -1, config.rms_norm_eps, bias=False
        )

        # tp是张量并行tensor parallel的缩写，单设备不用考虑。
        def _set_tp():
            def _set(layer, hint):
                layer.attrs["shard_strategy"] = hint

            # 注意力头的维度
            hd = config.head_dim
            # self.hidden_size = self.num_attention_heads * head_dim; (model\qwen2\qwen2_model.py#78)
            # 即hidden_size等于q，而k和v则等于 num_key_value_heads * head_dim
            q = self.self_attn.num_attention_heads * hd
            k = self.self_attn.num_key_value_heads * hd
            v = self.self_attn.num_key_value_heads * hd
            i = self.mlp.intermediate_size
            _set(
                self.self_attn.c_attn.weight,
                tp.ShardSingleDim("_shard_qkv_weight", dim=0, segs=[q, k, v]),
            )
            _set(
                self.self_attn.c_attn.bias,
                tp.ShardSingleDim("_shard_qkv_bias", dim=0, segs=[q, k, v]),
            )
            _set(self.self_attn.o_proj.weight, tp.ShardSingleDim("_shard_o", dim=1))
            _set(
                self.mlp.gate_up_proj.weight, tp.ShardSingleDim("_shard_mlp_up", segs=[i, i], dim=0)
            )
            _set(self.mlp.down_proj.weight, tp.ShardSingleDim("_shard_mlp_down", dim=1))

        self.tensor_parallel_shards = config.tensor_parallel_shards
        _set_tp()

    # (a)RMSNorm, Attention, (b)res(c=b+a), (c)RMSNorm, MLP, (d)res(e=d+c)
    # 残差链接 res: 叠加RMSNorm之前的数据，可以缓解梯度消失 和 保留原始信息。
    #              保留原始信息：注意力机制主要是对输入信息进行重新加权和聚焦，但在这个过程中可能会丢失一些原始输入中的有用信息。
    #                           通过叠加之前的输入，能够确保原始信息的一部分得以保留，这有助于模型在后续的处理中更好地利用这些信息
    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        out = self.input_layernorm(hidden_states)
        out = self.self_attn(out, paged_kv_cache, layer_id)
        hidden_states = self._apply_residual(out, residual=hidden_states)
        out = self.post_attention_layernorm(hidden_states)
        out = self.mlp(out)
        hidden_states = self._apply_residual(out, residual=hidden_states)
        return hidden_states

    # 残差链接，直接叠加原始输入
    def _apply_residual(self, out, residual):
        if self.tensor_parallel_shards > 1:
            return op.ccl_allreduce(out, "sum") + residual
        return out + residual

# 一个主类QWen2LMHeadModel里会有一个QWen2Model。
# 里面包含有 一个词嵌入层 + 多个QWen2DecoderLayer + 一个RMSNorm。 其中QWen2DecoderLayer是有num_hidden_layers个重复一样的。
# 推理时，输入数据循环通过每个 QWen2DecoderLayer，最后计算一次均方根归一化。
# * 词嵌入层在推理时不调用？CJM_TODO
class QWen2Model(nn.Module):
    def __init__(self, config: QWen2Config):
        self.embed_tokens = Qwen2Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [QWen2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)

    def forward(self, inputs: Tensor, paged_kv_cache: PagedKVCache):
        hidden_states = inputs
        for layer_id, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, paged_kv_cache, layer_id)
        hidden_states = self.norm(hidden_states)
        return hidden_states

# 千问2模型最外层类，模型加载时，创建的就是QWen2LMHeadModel
# 里面包含有一个 QWen2Model，如果tie_word_embeddings为False（词嵌入层和输出层不共享权重），还需要多加一个nn.Linear。
# * lm_head的forward 和 embed_tokens.lm_head_forward 计算是完全一样的。
#   如果 词嵌入层和输出层共享权重，则 embed_tokens 在 QWen2Model 的推理里会调用一次，最后输出层会再重复调用一次，两次计算使用的权重是同一份。
#   反之 不共享权重，则 embed_tokens 在 QWen2Model 的推理里会调用一次，最后输出层要额外多用一个nn.Linear来推理，其权重跟embed_tokens的不一致。
#   即tie_word_embeddings == False时，权重加载阶段会额外加载输出层的权重。
class QWen2LMHeadModel(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: QWen2Config):
        self.model = QWen2Model(config)
        self.tie_word_embeddings = config.tie_word_embeddings
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.dtype = config.dtype
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.intermediate_size = config.intermediate_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.rms_norm_eps = config.rms_norm_eps
        self.rope_theta = config.rope_theta
        self.vocab_size = config.vocab_size
        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.head_dim = config.head_dim

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def batch_forward(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        logit_positions: Optional[Tensor] = None,
    ):
        op_ext.configure()

        hidden_states = self.model(input_embeds, paged_kv_cache)
        # 按perfill的batch tokens的位置信息，调整输出特征的顺序。
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)

        if self.tie_word_embeddings:
            logits = self.model.embed_tokens.lm_head_forward(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def embed(self, input_ids: Tensor):
        if self.tensor_parallel_shards > 1:
            input_ids = op.ccl_broadcast_from_worker0(input_ids)
        return self.model.embed_tokens(input_ids)

    def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        def _index(x: te.Tensor):  # x[:-1,:]
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.model(input_embed, paged_kv_cache)
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        if self.tie_word_embeddings:
            logits = self.model.embed_tokens.lm_head_forward(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, paged_kv_cache

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        hidden_states = self.model(input_embed, paged_kv_cache)
        if self.tie_word_embeddings:
            logits = self.model.embed_tokens.lm_head_forward(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, paged_kv_cache

    # 在batch_perfill中，需要额外输入logit_positions，用于记录每个 token 在输入序列中的位置信息。
    # 在decode阶段，模型是逐个生成 token，本身就带有了每个token的位置信息，所以不需要显式地通过 logit_positions 来处理位置信息。
    # batch_decode是多个输入同时推理而组成的batch，对于一条输入而言，跟batch_size为1的decode是一样的，所以也不需要。
    # 而bacth_perfill是一条输入的多个tocken组成的batch。在 QWen2Model 推理结束后，先通过logit_positions调整输出特征顺序，再进入输出层得到最终结果。
    def batch_prefill(
        self, input_embeds: Tensor, logit_positions: Tensor, paged_kv_cache: PagedKVCache
    ):
        if self.tensor_parallel_shards > 1:
            logit_positions = op.ccl_broadcast_from_worker0(logit_positions)
        logits = self.batch_forward(input_embeds, paged_kv_cache, logit_positions)
        return logits, paged_kv_cache

    def batch_decode(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        logits = self.batch_forward(input_embeds, paged_kv_cache)
        return logits, paged_kv_cache

    def batch_verify(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        logits = self.batch_forward(input_embeds, paged_kv_cache)
        return logits, paged_kv_cache

    def create_paged_kv_cache(  # pylint: disable=too-many-arguments
        self,
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        prefill_chunk_size: tir.Var,
        page_size: tir.Var,
        support_sliding_window: tir.Var,
    ) -> PagedKVCache:
        return PagedKVCache.create_generic(
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=support_sliding_window,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads // self.tensor_parallel_shards,
            num_key_value_heads=self.num_key_value_heads // self.tensor_parallel_shards,
            head_dim=self.head_dim,
            rope_mode=RopeMode.NORMAL,
            rope_scale=1,
            rope_theta=self.rope_theta,
            dtype=self.dtype,
        )

    def get_default_spec(self):
        mod_spec = {
            "embed": {
                "input_ids": nn.spec.Tensor(["seq_len"], "int32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "prefill": {
                "input_embed": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode": {
                "input_embed": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_decode": {
                "input_embeds": nn.spec.Tensor(["batch_size", 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_verify": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "create_paged_kv_cache": {
                "max_batch_size": int,
                "max_total_seq_len": int,
                "prefill_chunk_size": int,
                "page_size": int,
                "support_sliding_window": int,
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
