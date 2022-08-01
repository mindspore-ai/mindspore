mindspore.Tensor.addcdiv
========================

.. py:method:: mindspore.Tensor.addcdiv(x1, x2, value)

    逐元素执行x1除以x2，将结果乘以标量value并将其添加到输入。

    .. math::
        y[i] = input\_data[i] + value[i] * (x1[i] / x2[i])

    参数：
        - **x1** (Tensor) - 分子张量。
        - **x2** (Tensor) - 分母张量。
        - **value** (Tensor) - x1/x2的倍数。

    返回：
        Tensor，shape和数据类型与当前Tensor相同。