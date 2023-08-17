mindspore.ops.reverse_sequence
==============================

.. py:function:: mindspore.ops.reverse_sequence(x, seq_lengths, seq_dim, batch_dim=0)

    对输入序列进行部分反转。

    参数：
        - **x** (Tensor) - 输入需反转的数据，其数据类型支持包括bool在内的所有数值型。
        - **seq_lengths** (Tensor) - 指定反转长度，为一维向量，其数据类型为int32或int64。
        - **seq_dim** (int) - 指定反转的维度，此值为必填参数。
        - **batch_dim** (int) - 指定切片维度。默认值： ``0`` 。

    返回：
        Tensor，shape和数据类型与输入 `x` 相同。

    异常：
        - **TypeError** - `seq_dim` 或 `batch_dim` 不是int。
        - **ValueError** - 如果 :math:`len(seq\_lengths) != x.shape[batch\_dim]`。
        - **ValueError** - 如果 :math:`batch\_dim == seq\_dim`。
        - **ValueError** - 如果 :math:`seq\_dim < 0` 或 :math:`seq\_dim >= len(x.shape)`。
        - **ValueError** - 如果 :math:`batch\_dim < 0` 或 :math:`batch\_dim >= len(x.shape)`。
        - **RuntimeError** - 如果 `seq_lengths` 中的任意一个值小于0。
        - **RuntimeError** - 如果 `seq_lengths` 中的任意一个值大于 `x.shape[seq_dim]` 。
