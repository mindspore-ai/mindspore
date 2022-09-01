mindspore.ops.ReverseSequence
==============================

.. py:class:: mindspore.ops.ReverseSequence(seq_dim, batch_dim=0)

    对输入序列进行部分反转。

    参数：
        - **seq_dim** (int) - 指定反转的维度，此值为必填参数。
        - **batch_dim** (int) - 指定切片维度。默认值：0。

    输入：
        - **x** (Tensor) - 输入需反转的数据，其数据类型支持包括bool在内的所有数值型。
        - **seq_lengths** (Tensor) - 指定反转长度，为一维向量，其数据类型为int32或int64。

    输出：
        Tensor，shape和数据类型与输入 `x` 相同。

    异常：
        - **TypeError** - `seq_dim` 或 `batch_dim` 不是int。
        - **ValueError** -  `batch_dim` 大于或等于 `x` 的shape长度。