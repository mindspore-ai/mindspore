mindspore.ops.sign
===================

.. py:function:: mindspore.ops.sign(x)

    按sign公式逐元素计算输入Tensor。

    .. math::
        \text{out}_{i} = \begin{cases}
                          -1 & \text{x} < 0 \\
                           0 & \text{x} = 0 \\
                           1 & \text{x} > 0
                         \end{cases}

    参数：
        - **x** (Tensor) - 输入Tensor。

    返回：
        Tensor， `x` 的sign计算结果。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。
