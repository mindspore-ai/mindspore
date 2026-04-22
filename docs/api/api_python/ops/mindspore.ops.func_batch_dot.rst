mindspore.ops.batch_dot
=======================

.. py:function:: mindspore.ops.batch_dot(x1, x2, axes=None)

    计算 `x1` 和 `x2` 中的向量点积。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    .. note::
        `x1` 和 `x2` 的第零维表示batch数量， `x1` 和 `x2` 数据类型为float32且秩大于或等于2。

    .. math::
        output = x1[batch, :] · x2[batch, :]

    参数：
        - **x1** (Tensor) - 第一个batch向量。
        - **x2** (Tensor) - 第二个batch向量。
        - **axes** (Union[int, tuple(int), list(int)]) - 指定计算轴。默认 ``None`` 。

    返回：
        Tensor
