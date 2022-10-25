mindspore.Tensor.reverse_sequence
==================================

.. py:method:: mindspore.Tensor.reverse_sequence(seq_lengths, seq_dim=0, batch_dim=0)

    对输入序列进行部分反转。

    参数：
        - **seq_lengths** (Tensor) - 指定反转长度，为一维向量，其数据类型为int32或int64。
        - **seq_dim** (int) - 指定反转的维度。默认值：0。
        - **batch_dim** (int) - 指定切片维度。默认值：0。

    返回：
        Tensor，shape和数据类型与输入 `x` 相同。

    异常：
        - **TypeError** - `seq_dim` 或 `batch_dim` 不是int。
        - **ValueError** -  `batch_dim` 大于或等于 `x` 的shape长度。