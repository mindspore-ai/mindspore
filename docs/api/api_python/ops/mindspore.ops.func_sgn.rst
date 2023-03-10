mindspore.ops.sgn
==================

.. py:function:: mindspore.ops.sgn(input)

    :func:`mindspore.ops.sign` 在复数上的扩展。
    对于实数输入，此方法与 :func:`mindspore.ops.sign` 一致。
    对于复数输入，此方法按照如下公式计算。

    .. math::
        \text{out}_{i} = \begin{cases}
                            0 & |\text{input}_i| == 0 \\
                            \frac{{\text{input}_i}}{|{\text{input}_i}|} & \text{otherwise}
                            \end{cases}

    参数：
        - **input** (Tensor) - 输入Tensor。

    返回：
        Tensor， `input` 的sgn计算结果。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。