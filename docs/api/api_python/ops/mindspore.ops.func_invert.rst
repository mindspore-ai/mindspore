mindspore.ops.invert
====================

.. py:function:: mindspore.ops.invert(x)

    对输入逐元素按位翻转。

    .. math::
        out_i = \sim x_{i}

    参数：
        - **x** (Tensor) - `x` 的shape为 :math:`(x_1, x_2, ..., x_R)`。其数据类型为int16或uint16。

    返回：
        Tensor，shape和类型与输入相同。

    异常：
        - **TypeError** - `x` 的数据类型不为int16或uint16。
