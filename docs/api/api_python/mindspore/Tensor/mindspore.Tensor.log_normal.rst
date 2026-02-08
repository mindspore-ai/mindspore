mindspore.Tensor.log_normal
============================

.. py:method:: mindspore.Tensor.log_normal(mean=1.0, std=2.0)

    使用给定均值 `mean` 和标准差 `std` 的对数正态分布的数值填充当前Tensor。

    .. math::
        \text{f}(x;1.0,2.0)=\frac{1}{x\delta \sqrt[]{2\pi} }e^{-\frac{(\ln x-\mu )^2}{2\delta ^2} }

    其中 :math:`\mu`、:math:`\delta` 分别是对数正态分布的均值和标准差。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **mean** (float, 可选) - 对数正态分布的均值。默认值： ``1.0``。
        - **std** (float, 可选) - 对数正态分布的标准差。默认值： ``2.0``。

    返回：
        Tensor，具有与当前Tensor相同的shape和dtype。