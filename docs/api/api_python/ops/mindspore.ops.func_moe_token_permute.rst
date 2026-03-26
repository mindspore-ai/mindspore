mindspore.ops.moe_token_permute
===============================

.. py:function:: mindspore.ops.moe_token_permute(tokens, indices, num_out_tokens=None, padded_mode=False)

    根据 `indices` 对 `tokens` 进行排列。具有相同索引的token将被分组在一起。

    .. warning::
        - `indices` 为二维的时候，第二维的size需要小于等于512。

    参数：
        - **tokens** (Tensor) - 输入的 token Tensor，将被进行排列。数据类型是bfloat16、float16或float32。shape为 :math:`(num\_tokens, hidden\_size)`，其中num_tokens和hidden_size是正整数。
        - **indices** (Tensor) - 用于排列token的索引Tensor。数据类型是int32或int64。shape为 :math:`(num\_tokens, topk)` 或 :math:`(num\_tokens,)`，其中num_tokens和topk是正整数。如果shape为后者，隐含topk为1。
        - **num_out_tokens** (int，可选) - 有效的输出token数量，当启用容量因子时，应等于未丢弃的token数量，应该大于等于0。默认 ``None`` ，表示不丢弃任何token。
        - **padded_mode** (bool，可选) - 如果为 ``True`` ，表示索引已填充，用于表示每个专家选择的token。目前只能设置为 ``False`` 。默认 ``False`` 。

    返回：
        tuple (Tensor)，表示2个Tensor组成的tuple，分别为排列后的 `tokens` 和排序后的 `indices` 。

        - **permuted_tokens** (Tensor) - 排列后的Tensor，数据类型与 `tokens` 相同。
        - **sorted_indices** (Tensor) - 索引Tensor，数据类型为int32，表示排列后的Tensor对应的索引。

    异常：
        - **TypeError** - 如果 `tokens` 或 `indices` 不是Tensor。
        - **TypeError** - 如果 `indices` 的数据类型不是int32或int64。
        - **TypeError** - 如果指定的 `num_out_tokens` 不是整数。
        - **TypeError** - 如果指定的 `padded_mode` 不是布尔值。
        - **ValueError** - 如果 `indices` 的第二维度大于512（如果存在）。
        - **ValueError** - 如果 `padded_mode` 设置为True。
        - **ValueError** - 如果 `tokens` 不是2维，或 `indices` 不是1维或2维的Tensor。
        - **RuntimeError** - 如果 `tokens` 和 `indices` 的第一维度不一致。
