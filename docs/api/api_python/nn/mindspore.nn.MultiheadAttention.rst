mindspore.nn.MultiheadAttention
========================================

.. py:class:: mindspore.nn.MultiheadAttention(embed_dim, num_heads, dropout=0., has_bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False)

    论文 `Attention Is All You Need <https://arxiv.org/pdf/1706.03762v5.pdf>`_ 中所述的多头注意力的实现。给定query向量，key向量和value，注意力计算流程如下：

    .. math::
        MultiHeadAttention(query, key, vector) = Concat(head_1, \dots, head_h)W^O

    其中， :math:`head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)` 。注意：输出层的投影计算中带有偏置参数。

    如果query、key和value相同，则上述即为自注意力机制的计算过程。

    参数：
        - **embed_dim** (int) - 模型的总维数。
        - **num_heads** (int) - 并行注意力头的数量。`num_heads` 需要能够被 `embed_dim` 整除（每个头的维数为 `embed_dim // num_heads`）。
        - **dropout** (float) - 应用到输入 `attn_output_weights` 上的随机丢弃比例. 默认值： ``0.0``。
        - **has_bias** (bool) - 是否给输入、输出投射层添加偏置。默认值： ``True``。
        - **add_bias_kv** (bool) - 是否给key、value序列的0维添加偏置。默认值： ``False``。
        - **add_zero_attn** (bool) - 是否给key、value序列的一维添加0。默认值： ``False``。
        - **kdim** (int) - key的总特征数。默认值： ``None`` （即 `kdim=embed_dim`）。
        - **vdim** (int) - value的总特征数。默认值：``None`` （即 `vdim=embed_dim`）。
        - **batch_first** (bool) - 如果为 ``True``，则输入输出Tensor的shape为 (batch, seq, feature)，否则shape为(seq, batch, feature)。 默认值： ``False`` 。

    输入：
        - **query** (Tensor) - Query矩阵。当输入非Batch数据时，Shape为： :math:`(L, E_q)` 。当输入Batch数据，参数 `batch_first=False` 时，Shape为 :math:`(L, N, E_q)` ，
          当 `batch_first=True` 时，Shape为 :math:`(N, L, E_q)`。其中， :math:`L` 为目标序列的长度， :math:`N` 为batch size，:math:`E_q` 为Query矩阵的维数 `embed_dim`。
          注意力机制通过Query与Key-Value运算以生成最终输出。详情请见："Attention Is All You Need"。
        - **key** (Tensor) - Key矩阵。当输入非Batch数据时，Shape为： :math:`(S, E_k)` 。当输入Batch数据，参数 `batch_first=False` 时，Shape为 :math:`(S, N, E_k)` ，
          当 `batch_first=True` 时，Shape为 :math:`(N, S, E_k)`。其中， :math:`S` 为源序列的长度， :math:`N` 为batch size，:math:`E_k` 为Key矩阵的维数 `kdim`。详情请见："Attention Is All You Need"。
        - **value** (Tensor) - Value矩阵。当输入非Batch数据时，Shape为： :math:`(S, E_v)` 。当输入Batch数据，参数 `batch_first=False` 时，Shape为 :math:`(S, N, E_v)` ，
          当 `batch_first=True` 时，Shape为 :math:`(N, S, E_v)`。其中， :math:`S` 为源序列的长度， :math:`N` 为batch size，:math:`E_v` 为Key矩阵的维数 `vdim`。详情请见："Attention Is All You Need"。
        - **key_padding_mask** (Tensor, optional) - 如果指定此值，则表示Shape为 :math:`(N, S)`的掩码将被用于 `key`。当输入非Batch数据时，Shape为： :math:`(S)` 。
          如果输入Tensor为Bool类型，则 `key` 中对应为 ``True`` 的位置将在Attention计算时被忽略。如果输入Tensor为Float类型，则将直接与 `key` 相加。默认值：``None``。
        - **need_weights** (bool) - 是否需要返回 `attn_output_weights`，如果为 ``True``，则输出包含 `attn_output_weights`。默认值：``True``。
        - **attn_mask** (Tensor, optional) - 如果指定此值，则表示Shape为 :math:`(L, S)` 或 :math:`(N\cdot\text{num\_heads}, L, S)` 的掩码将被用于Attention计算。其中 :math:`N` 为batch size，
          :math:`L` 为目标序列长度，:math:`S` 为源序列长度。如果输入为2维矩阵，则将自动沿batch维广播至3维矩阵。若为3维矩阵，则允许沿batch维使用不同的掩码。如果输入Tensor为Bool类型，则值为 ``True`` 对应位置允许被注意力计算。如果输入Tensor为Float类型，则将直接与注意力权重相加。默认值：``None``。
        - **average_attn_weights** (bool) - 如果为 ``True``， 则返回值 `attn_weights` 为注意力头的平均值。如果为 ``False``，则 ``attn_weights`` 分别返回每个注意力头的值。
          本参数仅在 `need_weights=True` 时生效。默认值： ``True`` 。

    输出：
        Tuple，表示一个包含(`attn_output`, `attn_output_weights`)的元组。

        - **attn_output** - 注意力机制的输出。当输入非Batch数据时，Shape为： :math:`(L, E)` 。当输入Batch数据， 参数 `batch_first=False` 时，Shape为 :math:`(L, N, E)` ，
          当 `batch_first=True` 时，Shape为 :math:`(N, L, E)`。其中， :math:`L` 为目标序列的长度， :math:`N` 为batch size， :math:`E` 为模型的总维数 `embed_dim`。
        - **attn_output_weights** - 仅当 ``need_weights=True`` 时返回。如果 `average_attn_weights=True`，则返回值 `attn_weights` 为注意力头的平均值。当输入非Batch数据时，
          Shape为： :math:`(L, S)` ，当输入Batch数据时，Shape为 :math:`(N, L, S)`。其中 :math:`N` 为batch size， :math:`L` 为目标序列的长度，:math:`S` 为源序列长度。
          如果 `average_attn_weights=False` ，分别返回每个注意力头的值。当输入非Batch数据时，Shape为： :math:`(\text{num\_heads}, L, S)` ，当输入Batch数据时，Shape为
          :math:`(N, \text{num\_heads}, L, S)`。
