mindspore.ops.bitwise_left_shift
=================================

.. py:function:: mindspore.ops.bitwise_left_shift(input, other)

    对输入 `input` 进行左移 `other` 位运算。

    .. math::

        \begin{aligned}
        &out_{i} =input_{i} << other_{i}
        \end{aligned}

    参数：
        - **input** (Union[Tensor, Scalar]) - 被左移的输入。
        - **other** (Union[Tensor, Scalar]) - 左移的位数。

    返回：
        Tensor，左移位运算后的结果。

    异常：
        - **TypeError** - `input` 或 `other` 都不是Tensor。
        - **TypeError** - `input` 或 `other` 不是int、int类型的Tensor或uint类型的Tensor。

