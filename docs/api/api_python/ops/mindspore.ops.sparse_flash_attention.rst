mindspore.ops.sparse_flash_attention
====================================

.. py:function:: mindspore.ops.sparse_flash_attention(query, key, value, sparse_indices, scale_value, *, block_table=None, actual_seq_lengths_query=None, actual_seq_lengths_kv=None, query_rope=None, key_rope=None, sparse_block_size=1, layout_query='BSND', layout_kv='BSND', sparse_mode=3, pre_tokens=2^63-1, next_tokens=2^63-1, attention_mode=0, return_softmax_lse=False) -> Tuple[Tensor, Tensor, Tensor]

    针对大序列长度推理场景的稀疏注意力计算模块。该模块通过"只计算关键部分"大幅减少计算量，基于稀疏索引选取重要的 Key 和 Value 进行注意力计算。
    以下为公式与参数 shape 中维度的含义：

    - B：batch维。
    - S1： `query` 即公式里的 `Q` 的序列长度。
    - S2： `key` 和 `value` 即公式里的 `K` 和 `V` 的序列长度。
    - N1： `query` 的多头数量，支持 1/2/4/8/16/32/64/128。
    - N2： `key` 和 `value` 的多头数量，仅支持 1。
    - D： `query` / `key` / `value` 的头维度，固定为 512。
    - D_rope： `query_rope` / `key_rope` 的头维度，固定为 64。

    .. note::
        - **attention_mode 约束**：当前仅支持 attention_mode=2（MLA-absorb模式）。使用 attention_mode=2 时，必须同时提供 `query_rope` 和 `key_rope`。
        - **return_softmax_lse 约束**：True表示返回，但图模式下不支持，False表示不返回；默认值为False。该参数仅在训练且 `layout_kv` 不为 ``"PA_BSND"`` 场景支持。
        - **layout 约束**：`layout_kv` 为 ``"PA_BSND"`` 时，`layout_query` 和 `layout_kv` 无需一致；`layout_kv` 为 ``"BSND"`` 或 ``"TND"`` 时，`layout_query` 和 `layout_kv` 需保持一致。

    sparse_flash_attention的计算公式定义如下：

    .. math::
        \text{attention_out} = \text{softmax}(\frac{Q \cdot \tilde{K}^T}{\sqrt{d_k}}) \cdot \tilde{V}

    其中 :math:`\tilde{K}, \tilde{V}` 为通过特定选取算法（如 `lightning_indexer`）选取的重要性较高的 Key 和 Value，
    通常具有稀疏或块稀疏特性。:math:`d_k` 为 :math:`Q, \tilde{K}` 的头维度。

    .. warning::
        这是一个实验性API，可能会发生变更或被删除。

    参数：
        - **query** (Tensor) - query Tensor，对应公式中的 `Q`。支持的数据类型为 float16 与 bfloat16。当 `layout_query` 为 ``"BSND"`` 时shape为 :math:`(B, S1, N1, D)`，当 `layout_query` 为 ``"TND"`` 时shape为 :math:`(T1, N1, D)`。N1 支持 1/2/4/8/16/32/64/128。
        - **key** (Tensor) - key Tensor，对应公式中的 `K`。数据类型与 `query` 一致。当 `layout_kv` 为 ``"BSND"`` 时shape为 :math:`(B, S2, N2, D)`，当 `layout_kv` 为 ``"TND"`` 时shape为 :math:`(T2, N2, D)`，当 `layout_kv` 为 ``"PA_BSND"`` 时shape为 :math:`(block\_num, block\_size, N2, D)`。N2 仅支持 1。
        - **value** (Tensor) - value Tensor，对应公式中的 `V`，数据类型以及 shape 与 `key` 一致。
        - **sparse_indices** (Tensor) - 稀疏索引Tensor，用于离散KV缓存访问。数据类型为 int32。当 `layout_query` 为 ``"BSND"`` 时shape为 :math:`(B, Q\_S, N2, sparse\_size)`，当 `layout_query` 为 ``"TND"`` 时shape为 :math:`(Q\_T, N2, sparse\_size)`。`sparse_size` 为一次离散选取的block数，必须大于0，且不超过 :math:`S2 / sparse\_block\_size` 向上取整后的值。
        - **scale_value** (float) - 缩放系数，用作query和key矩阵乘法后的乘法标量，通常为 1.0 / (D ** 0.5)。

    关键字参数：
        - **block_table** (Tensor，可选) - PageAttention场景下的块映射表。数据格式为 ND，数据类型为 int32，shape为 :math:`(B, block\_num)`，其中 `block_num` 不小于所有batch中最大的S2对应的block数量，即 :math:`S2_{max} / block\_size` 向上取整。默认值为： ``None`` 。
        - **actual_seq_lengths_query** (Tensor，可选) - 每个batch中query的有效token数量。数据格式为 ND，数据类型为 int32，shape为 :math:`(B,)`。用于变长序列的前缀和格式。默认值为： ``None`` 。
        - **actual_seq_lengths_kv** (Tensor，可选) - 每个batch中key和value的有效token数量。数据格式为 ND，数据类型为 int32，shape为 :math:`(B,)`。用于变长序列的前缀和格式。默认值为： ``None`` 。
        - **query_rope** (Tensor，可选) - MLA结构中query的RoPE信息。shape与query相同，但最后一维D替换为D_rope（64）。**当 attention_mode=2（MLA-absorb模式）时必须提供。** 默认值为： ``None`` 。
        - **key_rope** (Tensor，可选) - MLA结构中key的RoPE信息。shape与key相同，但最后一维D替换为D_rope（64）。**当 attention_mode=2（MLA-absorb模式）时必须提供。** 默认值为： ``None`` 。
        - **sparse_block_size** (int，可选) - 稀疏阶段的块大小，用于重要性分数计算。取值范围为 [1, 128]，必须是2的幂次方；当 `layout_kv` 为 ``"PA_BSND"`` 时，还需整除 `block_size`。不建议 `sparse_indices` 最后一维（即一次离散选取的block数）与 `sparse_block_size` 的乘积超过 100k，由于硬件限制可能会导致 OOM。默认值为： ``1`` 。
        - **layout_query** (str，可选) - query的数据布局格式。支持 ``"BSND"`` 和 ``"TND"``。默认值为： ``"BSND"``。
        - **layout_kv** (str，可选) - key的数据布局格式。支持 ``"TND"``、``"BSND"`` 和 ``"PA_BSND"``。为 ``"PA_BSND"`` 时，无需与 `layout_query` 取值保持一致；为 ``"BSND"`` 或 ``"TND"`` 时，需要与 `layout_query` 取值保持一致。默认值为： ``"BSND"``。
        - **sparse_mode** (int，可选) - 稀疏模式。仅支持 ``0`` （全部计算）与 ``3`` （rightDownCausal 掩码模式）。默认值为： ``3``。
        - **pre_tokens** (int，可选) - 稀疏计算中向前计算的token数量。当前仅支持 ``2^63-1`` 。默认值为： ``2^63-1`` 。
        - **next_tokens** (int，可选) - 稀疏计算中向后计算的token数量。当前仅支持 ``2^63-1`` 。默认值为： ``2^63-1`` 。
        - **attention_mode** (int，可选) - 注意力模式。当前仅支持 ``2`` （MLA-absorb 模式），即计算过程中将 query 和 key 的 nope 部分分别与 query_rope 和 key_rope 的 rope 部分沿头维度（D）拼接。使用该模式时必须同时提供 `query_rope` 和 `key_rope`。默认值为： ``0``。
        - **return_softmax_lse** (bool，可选) - 是否返回有效的 softmax_max 和 softmax_sum。为 ``True`` 时返回，但图模式下不支持；为 ``False`` 时不返回。该参数仅在训练且 `layout_kv` 不为 ``"PA_BSND"`` 时支持。默认值为： ``False``。

    返回：
        包含3个Tensor的元组。

        - **attention_out** (Tensor) - 注意力输出，数据类型与 `query` 一致。当 `layout_query` 为 ``"BSND"`` 时shape为 :math:`(B, S1, N1, D)`，当 `layout_query` 为 ``"TND"`` 时shape为 :math:`(T1, N1, D)`。
        - **softmax_max** (Tensor) - softmax中的最大值，数据类型为 float32。当 `layout_query` 为 ``"BSND"`` 时shape为 :math:`(B, N2, S1, N1/N2)`，当 `layout_query` 为 ``"TND"`` 时shape为 :math:`(N2, T1, N1/N2)`。
        - **softmax_sum** (Tensor) - softmax中的求和值，数据类型为 float32。shape与 `softmax_max` 一致。

    异常：
        - **TypeError** - `query` 不是 Tensor。
        - **TypeError** - `key` 不是 Tensor。
        - **TypeError** - `value` 不是 Tensor。
        - **TypeError** - `sparse_indices` 不是 Tensor。
        - **TypeError** - `query`、`key`、`value` 的数据类型不一致。
        - **TypeError** - `query`、`key`、`value` 的数据类型不是 float16 或 bfloat16。
        - **RuntimeError** - `layout_query` 不是 ``"BSND"`` 或 ``"TND"``。
        - **RuntimeError** - `layout_kv` 不是 ``"BSND"``、``"TND"`` 或 ``"PA_BSND"``。
        - **RuntimeError** - `sparse_block_size` 不在范围 [1, 128] 内或不是2的幂次方。
