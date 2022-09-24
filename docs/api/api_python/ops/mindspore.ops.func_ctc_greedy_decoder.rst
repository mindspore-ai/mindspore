mindspore.ops.ctc_greedy_decoder
================================

.. py:function:: mindspore.ops.ctc_greedy_decoder(inputs, sequence_length, merge_repeated=True)

    对输入中给定的logits执行贪婪解码。

    参数：
        - **inputs** (Tensor) - shape: :math:`(max\_time, batch\_size, num\_classes)`，数据类型必须是float32或者float64。`num_classes` 为 `num_labels + 1` classes，其中 `num_labels` 表示实际标签的个数，空标签默认使用 `num_classes - 1`。
        - **sequence_length** (Tensor) - shape: :math:`(batch\_size, )`，数据类型必须是int32，并且Tensor中的数值必须小于等于 `max_time`。
        - **merge_repeated** (bool) - True表示返回的结果中会合并重复的类。默认值为True。

    返回：
        - **decoded_indices** (Tensor) - shape: :math:`(total\_decoded\_outputs, 2)`，数据类型为int64。
        - **decoded_values** (Tensor) - shape: :math:`(total\_decoded\_outputs, )`，数据类型为int64。
        - **decoded_shape** (Tensor) - shape: :math:`(batch\_size, max\_decoded\_length)`，数据类型为int64。
        - **log_probability** (Tensor) - shape: :math:`(batch\_size, 1)`，包含序列的对数概率，其数据类型与 `inputs` 保持一致。

    异常：
        - **TypeError** - `merge_repeated` 不是一个布尔值。
        - **ValueError** - `inputs` 的shape长度不等于3。
        - **ValueError** - `sequence_length` 的shape长度不等于1。
        - **ValueError** - `sequence_length` 中的数值大于 `max_time`。
