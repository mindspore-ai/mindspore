mindspore.ops.sparse_flash_attention
====================================

.. py:function:: mindspore.ops.sparse_flash_attention(query, key, value, sparse_indices, scale_value, *, block_table=None, actual_seq_lengths_query=None, actual_seq_lengths_kv=None, query_rope=None, key_rope=None, sparse_block_size=1, layout_query='BSND', layout_kv='BSND', sparse_mode=3, pre_tokens=9223372036854775807, next_tokens=9223372036854775807, attention_mode=0, return_softmax_lse=False)

    针对大序列长度推理场景的稀疏注意力计算模块。该模块通过"只计算关键部分"大幅减少计算量，基于稀疏索引选取重要的Key和Value进行注意力计算。

    - B：batch维。
    - S1： `query` 的序列长度。
    - S2： `key` 和 `value` 的序列长度。
    - N1： `query` 的多头数量，支持 1/2/4/8/16/32/64/128。
    - N2： `key` 和 `value` 的多头数量，仅支持 1。
    - D： `query` / `key` / `value` 的头维度，固定为 512。
    - D_rope： `query_rope` / `key_rope` 的头维度，固定为 64。

    .. note::
        - **attention_mode 约束**：当前仅支持 attention_mode=2（MLA-absorb模式）。使用 attention_mode=2 时，必须同时提供 query_rope 和 key_rope。使用 attention_mode=0 或其他值不受支持，会导致错误。
        - **return_softmax_lse 约束**：如果需要调用反向（梯度计算）或需要有效的 softmax_max/softmax_sum 输出，必须设置为 True。

    sparse_flash_attention的计算公式定义如下：

    .. math::
        \text{attention\_out} = \text{softmax}(\frac{Q \cdot \tilde{K}^T}{\sqrt{d_k}}) \cdot \tilde{V}

    其中 :math:`\tilde{K}, \tilde{V}` 为基于 sparse_indices 选取的重要性较高的 Key 和 Value。

    .. warning::
        - 这是一个实验性API，可能会发生变更或被删除。
        - 只支持 `Atlas A2/A3` 推理系列产品。

    参数：
        - **query** (Tensor) - query Tensor，shape支持：当 `layout_query` 为 "BSND" 时为 :math:`(B, S1, N1, D)`，当 `layout_query` 为 "TND" 时为 :math:`(T1, N1, D)` 。支持的数据类型为float16以及bfloat16。
        - **key** (Tensor) - key Tensor，数据类型与 `query` 一致。当 `layout_kv` 为 "BSND" 时shape为 :math:`(B, S2, N2, D)`，当 `layout_kv` 为 "TND" 时shape为 :math:`(T2, N2, D)`，当 `layout_kv` 为 "PA_BSND" 时shape为 :math:`(block\_num, block\_size, N2, D)` 。
        - **value** (Tensor) - value Tensor，数据类型以及shape与 `key` 一致。
        - **sparse_indices** (Tensor) - 稀疏索引Tensor，用于离散KV缓存访问。数据类型为int32。当 `layout_query` 为 "BSND" 时shape为 :math:`(B, Q\_S, N2, sparse\_size)`，当 `layout_query` 为 "TND" 时shape为 :math:`(Q\_T, N2, sparse\_size)` 。 `sparse_size` 必须大于 0。
        - **scale_value** (float) - 缩放系数，用作query和key矩阵乘法后的乘法标量，通常为 1.0 / (D ** 0.5)。

    关键字参数：
        - **block_table** (Tensor，可选) - PageAttention场景下的块映射表。数据类型为int32，shape为2D，第一维为B。默认值为： ``None`` 。
        - **actual_seq_lengths_query** (Tensor，可选) - 每个batch中query的有效token数量。数据类型为int32，shape为 :math:`(B,)` 。用于变长序列的前缀和格式。默认值为： ``None`` 。
        - **actual_seq_lengths_kv** (Tensor，可选) - 每个batch中key和value的有效token数量。数据类型为int32，shape为 :math:`(B,)` 。用于变长序列的前缀和格式。默认值为： ``None`` 。
        - **query_rope** (Tensor，可选) - MLA结构中query的RoPE信息。shape与query相同，但最后一维D替换为D_rope（64）。**当 attention_mode=2（MLA-absorb模式）时必须提供。** 默认值为： ``None`` 。
        - **key_rope** (Tensor，可选) - MLA结构中key的RoPE信息。shape与key相同，但最后一维D替换为D_rope（64）。**当 attention_mode=2（MLA-absorb模式）时必须提供。** 默认值为： ``None`` 。
        - **sparse_block_size** (int，可选) - 稀疏阶段的块大小，用于重要性分数计算。取值范围为 [1, 128]，必须是2的幂次方。默认值为： ``1`` 。
        - **layout_query** (str，可选) - query的数据布局格式。支持 "BSND" 和 "TND"。默认值为： ``"BSND"`` 。
        - **layout_kv** (str，可选) - key的数据布局格式。支持 "TND"、"BSND" 和 "PA_BSND"。默认值为： ``"BSND"`` 。
        - **sparse_mode** (int，可选) - 稀疏模式。0表示全部计算，3表示rightDownCausal掩码模式。默认值为： ``3`` 。
        - **pre_tokens** (int，可选) - 稀疏计算中向前计算的token数量。默认值为： ``2^63-1`` 。
        - **next_tokens** (int，可选) - 稀疏计算中向后计算的token数量。默认值为： ``2^63-1`` 。
        - **attention_mode** (int，可选) - 注意力模式。当前仅支持传入2，表示MLA-absorb模式，即计算过程中会将query和key的nope部分分别和query_rope和key_rope的rope部分沿头维度（D）拼接。使用时必须同时提供query_rope和key_rope。默认值为： ``0`` 。
        - **return_softmax_lse** (bool，可选) - 是否返回有效的softmax_max和softmax_sum。如果需要有效的softmax_max/softmax_sum输出或需要调用反向（梯度计算），必须设置为 ``True`` 。当设置为 ``False`` 时，softmax_max和softmax_sum输出可能无效。默认值为： ``False`` 。
  
    返回：
        包含3个Tensor的元组：

        - **attention_out** (Tensor) - 注意力输出，数据类型与 `query` 一致。当 `layout_query` 为 "BSND" 时shape为 :math:`(B, S1, N1, D)`，当 `layout_query` 为 "TND" 时shape为 :math:`(T1, N1, D)` 。
        - **softmax_max** (Tensor) - softmax中的最大值，数据类型为float32。当 `layout_query` 为 "BSND" 时shape为 :math:`(B, N2, S1, N1/N2)`，当 `layout_query` 为 "TND" 时shape为 :math:`(N2, T1, N1/N2)` 。
        - **softmax_sum** (Tensor) - softmax中的求和值，数据类型为float32。shape与 `softmax_max` 一致。

    异常：
        - **TypeError** - `query` 不是Tensor。
        - **TypeError** - `key` 不是Tensor。
        - **TypeError** - `value` 不是Tensor。
        - **TypeError** - `sparse_indices` 不是Tensor。
        - **TypeError** - `query` 、 `key` 、 `value` 的数据类型不一致。
        - **TypeError** - 数据类型不是float16或bfloat16。
        - **RuntimeError** - `layout_query` 不是 "BSND" 或 "TND"。
        - **RuntimeError** - `layout_kv` 不是 "BSND"、"TND" 或 "PA_BSND"。
        - **RuntimeError** - `sparse_block_size` 不在范围 [1, 128] 内或不是2的幂次方。

    支持平台：
        ``Ascend``
