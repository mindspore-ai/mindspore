mindspore.ops.nextafter
=======================

.. py:function:: mindspore.ops.nextafter(input, other)

    逐元素计算 `input` 指向 `other` 的下一个可表示浮点值。

    .. math::
        out_i = \begin{cases}
            & input_i + eps, & \text{if } input_i < other_i \\
            & input_i - eps, & \text{if } input_i > other_i \\
            & input_i, & \text{if } input_i = other_i
        \end{cases}

    其中eps为输入tensor数据类型最小可表示增量值。

    参数：
        - **input** (Tensor) - 第一个输入tensor。
        - **other** (Tensor) - 第二个输入tensor。

    返回：
        Tensor
