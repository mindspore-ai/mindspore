mindspore.ops.lightning_indexer
===============================

.. py:function:: mindspore.ops.lightning_indexer(query, key, weights, *, actual_seq_lengths_query=None, actual_seq_lengths_key=None, block_table=None, layout_query="BSND", layout_key="BSND", sparse_count=2048, sparse_mode=3, pre_tokens=2^63-1, next_tokens=2^63-1, return_value=False) -> Tuple[Tensor, Tensor]

    本算子基于 DeepSeek Sparse Attention（DSA）算法，为每个 query token 计算 Top-k 稀疏索引。

    该算子是 DeepSeek-V3.2 中引入的 DSA 机制的核心组件，通过选择性地关注最相关的 token 而非处理整个输入序列，实现高效的长上下文建模。

    .. math::
        Indices = \text{Top-}k\left\{[1]_{1\times g} @ \left[(W @ [1]_{1\times S_k}) \odot
        \text{ReLU}\left(Q_{index} @ K_{index}^T\right)\right]\right\}

    其中 :math:`Q_{index} \in \mathbb{R}^{g \times d}` 是索引 Query，:math:`K_{index} \in \mathbb{R}^{S_k \times d}` 是上下文索引 Key，:math:`W \in \mathbb{R}^{g \times 1}` 是 head 权重，:math:`g` 是 GQA 组大小，:math:`d` 是 head 维度，:math:`S_k` 是上下文长度。

    - B: Batch 大小。
    - S1: `query` 序列长度。
    - S2: `key` 序列长度。
    - T1: TND 布局下 `query` 的总 token 数（所有 batch 序列长度之和）。
    - T2: TND 布局下 `key` 的总 token 数。
    - N1: `query` 的 head 数。支持 64（当 `layout_query` 为 TND 时仅支持 64）。
    - N2: `key` 的 head 数。仅支持 1。
    - D: head 维度。仅支持 128。

    .. warning::
        - 仅支持 Atlas A2 推理系列产品（Ascend 平台）。

    .. note::
        - `query`、 `key` 和 `weights` 的数据类型必须一致。
        - 在非 PageAttention 场景下， `layout_key` 应与 `layout_query` 保持一致。

    参数：
        - **query** (Tensor) - 输入 `query` 张量，shape 为 :math:`(B, S1, N1, D)` 或 :math:`(T1, N1, D)`。支持数据类型： ``mindspore.bfloat16`` 、 ``mindspore.float16`` 。
        - **key** (Tensor) - 输入 `key` 张量，shape 为 :math:`(B, S2, N2, D)`、:math:`(T2, N2, D)` 或 PageAttention 场景下为 :math:`(block\_count, block\_size, N2, D)`。支持数据类型： ``mindspore.bfloat16`` 、 ``mindspore.float16`` 。
        - **weights** (Tensor) - 用于加权聚合的 head 权重张量，shape 为 :math:`(B, S1, N1)` 或 :math:`(T1, N1)`。数据类型需要与 `query` 一致。

    关键字参数：
        - **actual_seq_lengths_query** (Tensor，可选) - 每个 batch 中 `query` 的有效 token 数，shape 为 :math:`(B,)`。支持数据类型： ``mindspore.int32`` 。当 `layout_query` 为 ``TND`` 时该参数是必需的，表示序列长度的前缀和（累积和）。默认值： ``None`` 。
        - **actual_seq_lengths_key** (Tensor，可选) - 每个 batch 中 `key` 的有效 token 数，shape 为 :math:`(B,)`。支持数据类型： ``mindspore.int32`` 。当 `layout_key` 为 ``TND`` 或 ``PA_BSND`` 时该参数是必需的。默认值： ``None`` 。
        - **block_table** (Tensor，可选) - PageAttention KV 存储的块映射表，shape 为 :math:`(B, max\_blocks)`。支持数据类型： ``mindspore.int32`` 。当 `layout_key` 为 ``PA_BSND`` 时该参数是必需的。默认值： ``None`` 。
        - **layout_query** (str，可选) - 指定输入 `query` 的数据布局。支持 ``BSND`` 和 ``TND`` 。默认值： ``BSND`` 。
        - **layout_key** (str，可选) - 指定输入 `key` 的数据布局。支持 ``BSND`` 、 ``TND`` 和 ``PA_BSND`` 。默认值： ``BSND`` 。
        - **sparse_count** (int，可选) - 稀疏选择中保留的 Top-k token 数量。取值范围为 [1, 2048]。默认值： ``2048`` 。
        - **sparse_mode** (int，可选) - 指定稀疏模式。默认值： ``3`` 。

            - 0：defaultMask 模式。
            - 3：rightDownCausal 模式，对应右下角顶点划分的下三角场景。

        - **pre_tokens** (int，可选) - 稀疏计算的保留参数，表示向前计算多少 token。仅支持默认值。默认值： ``2^63-1`` 。
        - **next_tokens** (int，可选) - 稀疏计算的保留参数，表示向后计算多少 token。仅支持默认值。默认值： ``2^63-1`` 。
        - **return_value** (bool，可选) - 是否输出有效的 `sparse_values`。仅在 `layout_key` 不为 ``PA_BSND`` 时支持 ``True`` 。默认值： ``False`` 。

    返回：
        一个Tensor元组，包含 `sparse_indices` 和 `sparse_values` 。

        - `sparse_indices` 是 Top-k token 的索引，数据类型为 ``mindspore.int32`` ，shape 为 :math:`(B, S1, N2, sparse\_count)` 或 :math:`(T1, N2, sparse\_count)`。
        - `sparse_values` 是 Top-k token 对应的值，数据类型与 `query` 一致，shape 与 `sparse_indices` 相同。仅当 `return_value` 为 ``True`` 时有效。
