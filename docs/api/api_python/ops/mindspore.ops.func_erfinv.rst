mindspore.ops.erfinv
====================

.. py:function:: mindspore.ops.erfinv(input)

    计算输入的逆误差函数。逆误差函数在 `(-1, 1)` 范围内定义为：

    .. math::
        erfinv(erf(x)) = x

    其中 :math:`x` 代表输入Tensor。

    参数：
        - **input** (Tensor) - 输入Tensor，数据类型必须为float16、float32、float64。

    返回：
        Tensor，具有与 `input` 相同的数据类型和shape。

    异常：
        - **TypeError** - `input` 的数据类型不是float16、float32、float64。
