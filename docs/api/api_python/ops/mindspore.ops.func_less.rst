mindspore.ops.less
===================

.. py:function:: mindspore.ops.less(input, other)

    逐元素计算 :math:`input < other` 的值。

    .. math::
        out_{i} = \begin{cases}
            & \text{True,    if } input_{i}<other_{i} \\
            & \text{False,   if } input_{i}>=other_{i}
            \end{cases}

    .. note::
        支持隐式类型转换。

    参数：
        - **input** (Union[Tensor, Number, bool]) - 第一个输入。
        - **other** (Union[Tensor, Number, bool]) - 第二个输入。

    返回：
        Tensor
