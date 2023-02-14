mindspore.ops.sgn
==================

.. py:function:: mindspore.ops.sgn(x)

    :func:`mindspore.ops.sign` 在复数上的扩展。
    对于实数输入，此方法与 :func:`mindspore.ops.sign` 一致。
    对于复数输入，此方法按照如下公式计算。

    .. math::
        \text{out}_{i} = \begin{cases}
                            0 & |\text{x}_i| == 0 \\
                            \frac{{\text{x}_i}}{|{\text{x}_i}|} & \text{otherwise}
                            \end{cases}

    参数：
        - **x** (Tensor) - 输入Tensor。

    返回：
        Tensor， `x` 的sgn计算结果。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。