mindspore.ops.LogNormalReverse
==============================

.. py:class:: mindspore.ops.LogNormalReverse(mean=1.0, std=2.0)

    用给定均值和标准差初始化对数正态分布，并以此填充输入Tensor的元素。

    .. math::
        \text{f}(x;\mu,\delta)=\frac{1}{x\delta \sqrt[]{2\pi} }e^{-\frac{(\ln x-\mu )^2}{2\delta ^2} }

    其中 \mu， \delta 分别是对数正态分布的均值和标准差。

    参数：
        - **mean** (float，可选) - 正态分布的均值，float类型。默认值： ``1.0`` 。
        - **std** (float，可选) - 正态分布的标准差，float类型。默认值： ``2.0`` 。

    输入：
        - **input** (Tensor) - 要用对数正态分布生成的Tensor。必须是以下类型之一：float16、float32。

    输出：
        Tensor，与 `input` 的shape及数据类型相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **ValueError** - 如果 `input` 是NULL。
