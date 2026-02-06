mindspore.mint.greater_equal
==============================

.. py:function:: mindspore.mint.greater_equal(input, other) -> Tensor

    逐元素比较是否 :math:`input >= other` 。

    .. math::
        out_{i} =\begin{cases}
            & \text{True,    if } input_{i}>=other_{i} \\
            & \text{False,   if } input_{i}<other_{i}
            \end{cases}

    .. note::
        - 支持广播、隐式类型转换。
        - 输入必须是两个tensor（数据类型不能同时是bool），或一个tensor和一个scalar。

    参数：
        - **input** (Union[Tensor, Number]) - 第一个输入。
        - **other** (Union[Tensor, Number]) - 第二个输入。

    返回：
        Bool型Tensor
