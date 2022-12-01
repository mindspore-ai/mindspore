mindspore.ops.bitwise_right_shift
=================================

.. py:function:: mindspore.ops.bitwise_right_shift(x, other)

    对输入 `x` 进行右移 `other` 位运算。

    .. math::

        \begin{aligned}
        &out_{i} =x_{i} >> other_{i}
        \end{aligned}

    参数：
        - **x** (Union[Tensor, Scalar]) - 被右移的输入。
        - **other** (Union[Tensor, Scalar]) - 右移的位数。

    返回：
        Tensor，右移位运算后的结果。

    异常：
        - **TypeError** - `x` 或 `other` 都不是Tensor。
        - **TypeError** - `x` 或 `other` 不是int、int类型的Tensor或uint类型的Tensor。

