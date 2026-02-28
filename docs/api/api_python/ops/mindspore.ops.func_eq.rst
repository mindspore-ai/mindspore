mindspore.ops.eq
================

.. py:function:: mindspore.ops.eq(input, other)

    逐元素计算两个输入是否相等。

    .. math::
        out_{i} =\begin{cases}
            & \text{True,    if } input_{i} = other_{i} \\
            & \text{False,   if } input_{i} \ne other_{i}
            \end{cases}

    .. note::
        - 支持隐式类型转换。
        - 输入必须是两个Tensor，或是一个Tensor和一个Scalar。
        - 两个输入的shape支持广播。

    参数：
        - **input** (Tensor) - 第一个输入。
        - **other** (Union[Tensor, Number]) - 第二个输入。

    返回：
        Tensor
