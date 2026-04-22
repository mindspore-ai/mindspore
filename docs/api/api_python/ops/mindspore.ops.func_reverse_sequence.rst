mindspore.ops.reverse_sequence
==============================

.. py:function:: mindspore.ops.reverse_sequence(x, seq_lengths, seq_dim, batch_dim=0)

    对输入序列进行部分反转。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    参数：
        - **x** (Tensor) - 输入tensor。
        - **seq_lengths** (Tensor) - 指定反转长度。
        - **seq_dim** (int) - 指定反转的维度。
        - **batch_dim** (int) - 指定切片维度，默认 ``0`` 。

    返回：
        Tensor
