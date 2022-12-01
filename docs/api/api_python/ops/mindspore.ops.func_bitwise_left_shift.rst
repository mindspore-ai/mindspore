mindspore.ops.bitwise_left_shift
=================================

.. py:function:: mindspore.ops.bitwise_left_shift(x, other)

    对输入 `x` 进行左移 `other` 位运算。

    .. math::

        \begin{aligned}
        &out_{i} =x_{i} << other_{i}
        \end{aligned}

    参数：
        - **x** (Union[Tensor, Scalar]) - 被左移的输入。
        - **other** (Union[Tensor, Scalar]) - 左移的位数。

    返回：
        Tensor，左移位运算后的结果。

    异常：
        - **TypeError** - `x` 或 `other` 都不是Tensor。
        - **TypeError** - `x` 或 `other` 不是int、int类型的Tensor或uint类型的Tensor。

