mindspore.Tensor.erfinv
=======================

.. py:method:: mindspore.Tensor.erfinv()

    计算输入的逆误差函数。逆误差函数在 `(-1, 1)` 范围内定义为：

    .. math::
        erfinv(erf(x)) = x

    返回：
        Tensor，具有与当前Tensor相同的数据类型和shape。

    异常：
        - **TypeError** - 当前Tensor的数据类型不是float16、float32、float64。
