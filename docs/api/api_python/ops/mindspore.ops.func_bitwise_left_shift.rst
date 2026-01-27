mindspore.ops.bitwise_left_shift
=================================

.. py:function:: mindspore.ops.bitwise_left_shift(input, other)

    逐元素对输入 `input` 进行左移位运算，移动的位数由 `other` 指定。

    .. math::

        \begin{aligned}
        &out_{i} =input_{i} << other_{i}
        \end{aligned}

    参数：
        - **input** (Union[Tensor, int, bool]) - 被左移的输入tensor。
        - **other** (Union[Tensor, int, bool]) - 左移的位数。

    返回：
        Tensor
