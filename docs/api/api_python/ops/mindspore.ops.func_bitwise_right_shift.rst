mindspore.ops.bitwise_right_shift
=================================

.. py:function:: mindspore.ops.bitwise_right_shift(input, other)

    对输入 `input` 进行右移 `other` 位运算。

    .. math::

        \begin{aligned}
        &out_{i} =input_{i} >> other_{i}
        \end{aligned}

    参数：
        - **input** (Union[Tensor, Scalar]) - 被右移的输入。
        - **other** (Union[Tensor, Scalar]) - 右移的位数。

    返回：
        Tensor，右移位运算后的结果。

    异常：
        - **TypeError** - `input` 或 `other` 都不是Tensor。
        - **TypeError** - `input` 或 `other` 不是int、int类型的Tensor或uint类型的Tensor。

