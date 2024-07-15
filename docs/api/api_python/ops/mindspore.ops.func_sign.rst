mindspore.ops.sign
===================

.. py:function:: mindspore.ops.sign(input)

    按sign公式逐元素计算输入Tensor。注意：在输入为NaN且数据类型为float64时，该算子计算结果为NaN。

    .. math::
        \text{out}_{i} = \begin{cases}
                          -1 & \text{input}_{i} < 0 \\
                           0 & \text{input}_{i} = 0 \\
                           1 & \text{input}_{i} > 0
                         \end{cases}

    参数：
        - **input** (Tensor) - 输入Tensor。

    返回：
        Tensor， `input` 的sign计算结果。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
