"""Configuration dataclasses used in MLC LLM serving"""

import json
from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional, Tuple, Union


@dataclass
class EngineConfig:  # pylint: disable=too-many-instance-attributes
    """The class of MLCEngine execution configuration.

    Parameters
    ----------
    model : str
        The path to the model directory.

    model_lib : str
        The path to the model library.

    additional_models : List[Union[str, Tuple[str, str]]]
        The paths to the additional models' directories (and model libraries).
        Each element is a single string (denoting the model directory)
        or a tuple of two strings (denoting the model directory and model lib path).

    mode : Literal["local", "interactive", "server"]
        The engine mode in MLC LLM.
        We provide three preset modes: "local", "interactive" and "server".
        The default mode is "local".
        The choice of mode decides the values of "max_num_sequence", "max_total_sequence_length"
        and "prefill_chunk_size" when they are not explicitly specified.
        1. Mode "local" refers to the local server deployment which has low
        request concurrency. So the max batch size will be set to 4, and max
        total sequence length and prefill chunk size are set to the context
        window size (or sliding window size) of the model.
        2. Mode "interactive" refers to the interactive use of server, which
        has at most 1 concurrent request. So the max batch size will be set to 1,
        and max total sequence length and prefill chunk size are set to the context
        window size (or sliding window size) of the model.
        3. Mode "server" refers to the large server use case which may handle
        many concurrent request and want to use GPU memory as much as possible.
        In this mode, we will automatically infer the largest possible max batch
        size and max total sequence length.

        You can manually specify arguments "max_num_sequence", "max_total_sequence_length" and
        "prefill_chunk_size" to override the automatic inferred values.

    tensor_parallel_shards : Optional[int]
        Number of shards to split the model into in tensor parallelism multi-gpu inference.

    pipeline_parallel_stages : Optional[int]
        Number of pipeline stages to split the model layers for pipeline parallelism.

    gpu_memory_utilization : Optional[float]
        A number in (0, 1) denoting the fraction of GPU memory used by the server in total.
        It is used to infer to maximum possible KV cache capacity.
        When it is unspecified, it defaults to 0.85.
        Under mode "local" or "interactive", the actual memory usage may be
        significantly smaller than this number. Under mode "server", the actual
        memory usage may be slightly larger than this number.

    kv_cache_page_size : int
        在page kv cache中, 每个page处理的连续的token的数量.
        The number of consecutive tokens handled in each page in paged KV cache.

    max_num_sequence : Optional[int]
        KV缓存在任何时候允许处理的最大序列数, 涉及到显存分配.
        The maximum number of sequences that are allowed to be
        processed by the KV cache at any time.

    max_total_sequence_length : Optional[int]
        任何时候KV cache中允许存在KV数据的最大token总数
        The maximum total number of tokens whose KV data are allowed
        to exist in the KV cache at any time.

    max_single_sequence_length : Optional[int]
        单个序列允许的最大长度
        The maximum length allowed for a single sequence in the engine.

    prefill_chunk_size : Optional[int]
        prefill中的最大总序列长度。
        The maximum total sequence length in a prefill.

    sliding_window_size : Optional[int]
        滑动窗口注意力, 是一种注意力机制变体, 旨在降低计算复杂度并提高计算效率，同时在一定程度上保持模型的性能.
        * 注意力机制计算输入序列中 每个位置与其他所有位置之间的相关性。
          假设输入序列长度为, 对于每个位置, 都需要计算其与个位置(包括自身)的注意力权重, 这就导致了计算复杂度为O(n^2).
        * SWA机制中, 通过限制注意力的计算范围，将复杂度降低. 只计算每个位置与一个固定窗口内的位置之间的注意力权重.
          如窗口大小为k, 对于序列中的每个位置, 只需计算与周围k个位置(左右各k/2个位置,假设k为偶数)的注意力权重, 
          计算复杂度变为O(nk), 当时k远小于n时,计算量大幅减少。
          -> 提高计算效率, 能够很好地捕捉局部信息; 但会忽略长距离的依赖关系, 所以窗大小的选择也比较困难.
        The sliding window size in sliding window attention (SWA).

    attention_sink_size : Optional[int]
        注意力汇聚点(attention sinks), 需要指定保存了的sinks的数量, 目前仅支持Mistral, 默认为4.
        在注意力机制运行过程中，信息集中汇聚的位置或元素。在启用滑动窗口的情况下，由于窗口的限制，信息的流动和汇聚方式发生了变化。注意力汇聚点可能是窗口内对周围元素有强烈影响力的元素，或者是多个窗口之间信息传递的关键节点.
        The number of attention sinks when sliding window is enabled..

    max_history_size: Optional[int]
        RNN状态回滚的最大历史大小
        The maximum history size for RNN state to roll back.

    kv_state_kind: Optional[Literal["kv_cache", "rnn_state"]]
        The kind of cache.

    speculative_mode : Literal["disable", "small_draft", "eagle", "medusa"]
        推测性解码(speculative decoding), 在NLP领域用于加速文本生成过程的技术.
        核心思想是通过利用一个小而快速的辅助模型来对下一个可能的 token(词元)进行推测，从而减少主模型(通常是大型且计算成本高的模型)的计算量.
        The speculative mode.
        "disable" means speculative decoding is disabled.
        "small_draft" means the normal speculative decoding (small draft) mode.
        "eagle" means the eagle-style speculative decoding. (决策过程可能更具果敢性，如同鹰在捕猎时迅速锁定目标? 辅助模型可能基于更激进的策略来推测 token,减少在多种可能选择中的犹豫)
        "medusa" means the medusa-style speculative decoding.

    spec_draft_length : int
        在推测性解码中生成的token数量 (推测性解码是利用小模型快速推测以减少大模型的计算量, 而小模型推测的结果表示为draft?草稿?).
        设为0, 则为自适应的模式, 即其数量会随engine状态的变化被自动调整.
        The number of tokens to generate in speculative proposal (draft).
        Being 0 means to enable adaptive speculative mode, where the draft length
        will be automatically adjusted based on engine state.

    spec_tree_width : int
        推测树的宽度, 推测树是一种在推测性解码过程中用于组织和表示可能的解码路径的数据结构.
        每个节点代表一个在解码过程中的推测状态或一个可能的词元(token), 而边则表示从一个状态到下一个状态的过渡, 通常基于一定的概率或模型的决策规则。
        The width of the speculative decoding tree.

    prefix_cache_mode : Literal["disable", "radix"]
        Prefix Cache(前缀缓存), 是一种存储结构, 用于保存与输入文本前缀相关的信息.
        llm中给定一个输入文本序列, 模型会逐词(或逐token词元)生成后续文本。Prefix Cache 记录了模型在处理输入文本前缀时的中间状态、计算结果或相关的统计信息，以便在后续生成过程中快速检索和利用。
        prefix cache 存储的是与输入文本前缀相关的综合信息，范围更广; kv cache 聚焦于自注意力机制中的键值对.
        prefix cache 在文本生成过程中的作用更偏向于整体的引导和辅助; kv cache 主要在计算注意力权重的过程中发挥作用.
        prefix cache 和 kv cache 一起使用的例子:
        给定开头"在经济持续增长的背景下", 完成新闻续写。
        1. 缓存前缀信息 prefix cache:
           将给定的前缀文本 “在经济持续增长的背景下” 输入到模型中。模型会对这个前缀文本进行处理，包括词法分析、向量化等操作。
           这些处理后的前缀信息（如词向量、位置编码等相关数据）被存储到 Prefix Cache 中。
        2. 生成第一个单词(使用 Prefix Cache 和 KV Cache):
           模型开始生成下一个单词。此时，模型从 Prefix Cache 中提取前缀信息, 同时, 多头注意力机制开始计算注意力权重。
           对于第一次计算，由于没有历史生成的单词，所以 KV Cache 是空的。
           模型根据 Prefix Cache 中的前缀信息和当前计算的注意力权重(开始填充 KV Cache), 通过前馈神经网络等组件, 
           生成第一个单词 “,” (假设生成的第一个单词是逗号)。这个单词的键值对信息被存储到 KV Cache 中。
        3. 接着生成第二个单词。模型再次从 Prefix Cache 中提取前缀信息，同时从 KV Cache 中提取上一步生成的单词 “，” 的键值对信息。
           利用这些信息，多头注意力机制重新计算注意力权重(更新 KV Cache),然后通过前馈神经网络等，生成第二个单词 “各行各业”。这个单词的键值对也被存储到 KV Cache 中。
           以此类推, 在整个文本生成过程中, Prefix Cache 一直存储着原始的前缀信息, KV Cache 不断更新和存储每一步生成单词的键值对信息，帮助模型快速高效地生成完整的新闻文章, 比如“在经济持续增长的背景下,各行各业都迎来了新的机遇”。
        可共同减少重复计算量，但需要一定空间缓存信息。
        Paged Radix Tree(分页基数树), 一种压缩前缀树。基于分页，更好管理内存。
        The prefix cache mode.
        "disable" means no prefix cache is disabled.
        "radix" means the paged radix tree based prefix cache mode.

    prefix_cache_max_num_recycling_seqs: Optional[int]
        Prefix Cache是一种缓存策略, 就会涉及到缓存空间满了, 需要挪位置的问题。
        以 LRU (最近最少使用-Least Recently Used) 策略为例, 即先回收最久未被使用的数据。
        这里是指定最大的回收序列数量.
        The maximum number of recycling sequences in prefix cache, default as max_num_sequence.
        And set 0 to disable prefix cache, set -1 to have infinite capacity prefix cache.

    prefill_mode : Literal["chunked", "hybrid"]
        chunked: 指常规的chunked prefill, 即在prefill阶段 把输入文本划分为多个块, 每个块依次进行预填充操作 (普通的prefill是一次性全部并行计算, 内存要求高, 对计算资源分配也不合理)
        hybrid prefill: 多种不同的预填充策略组合在一起使用?
        split-fuse: 首先会对输入文本进行拆分操作, 拆分后的各个子文本经过独立处理（如通过预填充、编码等操作）后，会进行融合操作.
                    拆分可能基于多种因素，如文本长度、语义单元、主题相关性,也可能是优化资源分配.
                    在融合阶段，会根据子文本之间的逻辑关系和重要性，将它们的特征向量通过加权求和、拼接等方式融合在一起.?
        参数默认一般为"hybrid", 搜索发现该模式下, 会将解码请求添加到perfill输入中 (cpp\serve\engine_actions\batch_prefill_base.cc#228)
        The prefill mode.
        "chunked" means the basic prefill with chunked input enabled.
        "hybrid" means the hybrid prefill or split-fuse,
        so that decode step will be converted into prefill.

    verbose : bool
        A boolean indicating whether to print logging info in engine.
    """

    model: Optional[str] = None
    model_lib: Optional[str] = None
    additional_models: List[Union[str, Tuple[str, str]]] = field(default_factory=list)
    mode: Optional[Literal["local", "interactive", "server"]] = None
    tensor_parallel_shards: Optional[int] = None
    pipeline_parallel_stages: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None
    kv_cache_page_size: int = 16
    max_num_sequence: Optional[int] = None
    max_total_sequence_length: Optional[int] = None
    max_single_sequence_length: Optional[int] = None
    prefill_chunk_size: Optional[int] = None
    sliding_window_size: Optional[int] = None
    attention_sink_size: Optional[int] = None
    max_history_size: Optional[int] = None
    kv_state_kind: Optional[Literal["kv_cache", "rnn_state"]] = None
    speculative_mode: Literal["disable", "small_draft", "eagle", "medusa"] = "disable"
    spec_draft_length: int = 0
    spec_tree_width: int = 1
    prefix_cache_mode: Literal["disable", "radix"] = "radix"
    prefix_cache_max_num_recycling_seqs: Optional[int] = None
    prefill_mode: Literal["chunked", "hybrid"] = "hybrid"
    verbose: bool = True

    def asjson(self) -> str:
        """Return the config in string of JSON format."""
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(json_str: str) -> "EngineConfig":
        """Construct a config from JSON string."""
        return EngineConfig(**json.loads(json_str))
