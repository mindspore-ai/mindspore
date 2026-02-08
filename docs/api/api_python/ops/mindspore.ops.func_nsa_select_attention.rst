mindspore.ops.nsa_select_attention
==================================

.. py:function:: mindspore.ops.nsa_select_attention(query, key, value, topk_indices, scale_value, head_num, select_block_size, select_block_count, *, atten_mask=None, actual_seq_qlen, actual_seq_kvlen) -> Tuple[Tensor, Tensor, Tensor]

    本算子用于在训练场景中计算原生稀疏注意力（Native Sparse Attention）算法的选择性注意力机制。

    该算子实现了原生稀疏注意力中的选择性注意力计算，通过依据 `topk_indices` 选择特定的注意力块以高效计算注意力权重。

    .. warning::
        - 输入 `query`、`key` 和 `value` 的布局固定为 ``TND`` 。
        - 只支持 `Atlas A2` 训练系列产品。
        - `topk_indices` 若存在取值越界，可能导致未定义行为。

    参数：
        - **query** (Tensor) - 输入 `query` 张量，shape 为 :math:`(T_1, N_1, D_1)`，其中 :math:`T_1` 为序列长度， :math:`N_1` 为注意力头数， :math:`D_1` 为单头维度。支持数据类型：``mindspore.bfloat16`` 、``mindspore.float16`` 。支持非连续 Tensor，不支持空 Tensor。
        - **key** (Tensor) - 输入 `key` 张量，shape 为 :math:`(T_2, N_2, D_1)`，其中 :math:`T_2` 为 `key` 序列长度， :math:`N_2` 为 `key` 的头数， :math:`D_1` 为单头维度（与 `query` 相同）。支持数据类型：``mindspore.bfloat16`` 、``mindspore.float16`` 。支持非连续 Tensor，不支持空 Tensor。
        - **value** (Tensor) - 输入 `value` 张量，shape 为 :math:`(T_2, N_2, D_2)`，其中 :math:`T_2` 为 `value` 序列长度， :math:`N_2` 为 `value` 的头数， :math:`D_2` 为 `value` 的单头维度。支持数据类型：``mindspore.bfloat16`` 、``mindspore.float16`` 。支持非连续 Tensor，不支持空 Tensor。
        - **topk_indices** (Tensor) - 索引张量，shape 为 :math:`(T_1, N_2, select\_block\_count)`，用于指定选择哪些注意力块。支持数据类型：``mindspore.int32`` 。支持非连续 Tensor，不支持空 Tensor。对于每个 batch，`topk_indices` 的每个元素必须满足 :math:`0 \leq index \leq S_2 / 64`，其中 :math:`S_2` 为该 batch 的有效 KV 序列长度，``64`` 为 `select_block_size`。
        - **scale_value** (float) - 作用于注意力分数的缩放因子，通常设为 :math:`D^{-0.5}`，其中 :math:`D` 为头维度。
        - **head_num** (int) - 每设备的注意力头数量，应等于 `query` 的 :math:`N_1` 轴长度。
        - **select_block_size** (int) - 选择窗口的大小。目前仅支持 ``64`` 。
        - **select_block_count** (int) - 选择窗口的数量。当 `select_block_size` 为 ``64`` 时，该参数值应为 ``16`` 。

    关键字参数：
        - **atten_mask** (Tensor，可选) - 注意力掩码张量。目前不支持。默认： ``None`` 。
        - **actual_seq_qlen** (Union[tuple[int], list[int]]) - 每个 batch 中 `query` 对应的大小（前缀和模式），必须为非递减整数序列，最后一个值等于 :math:`T_1` 。
        - **actual_seq_kvlen** (Union[tuple[int], list[int]]) - 每个 batch 中 `key` 和 `value` 对应的大小（前缀和模式），必须为非递减整数序列，最后一个值等于 :math:`T_2` 。

    返回：
        一个Tensor元组，包含 `attention_out`、`softmax_max` 和 `softmax_sum` 。

        - `attention_out` 是注意力的输出结果。
        - `softmax_max` 是Softmax计算的中间最大值结果，用于反向计算。
        - `softmax_sum` 是Softmax计算的中间求和结果，用于反向计算。

    异常：
        - **TypeError** - `query`、`key`、`value` 或 `topk_indices` 不是 Tensor。
        - **TypeError** - `scale_value` 不是 float。
        - **TypeError** - `head_num`、`select_block_size` 或 `select_block_count` 不是 int。
        - **TypeError** - `actual_seq_qlen` 或 `actual_seq_kvlen` 在提供时不是 int 列表。
        - **RuntimeError** - `query`、`key` 与 `value` 的数据类型不一致。
        - **RuntimeError** - `query`、`key` 与 `value` 的 batch 大小不相等。
        - **RuntimeError** - `head_num` 与 `query` 的头维度不匹配。
        - **RuntimeError** - `topk_indices` 存在超出有效范围 :math:`0 \leq index \leq S_2 / 64` 的取值。
        - **RuntimeError** - 维度约束不满足：:math:`D_q == D_k` 且 :math:`D_k >= D_v`。
