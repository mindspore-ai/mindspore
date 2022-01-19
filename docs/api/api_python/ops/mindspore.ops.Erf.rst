mindspore.ops.Erf
=================

.. py:class:: mindspore.ops.Erf()

    逐元素计算 `x` 的高斯误差函数。

    .. math::

        erf(x)=\frac{2} {\sqrt{\pi}} \int\limits_0^{x} e^{-t^{2}} dt

    **输入：**

    - **x** (Tensor) - 高斯误差函数的输入Tensor。维度必须小于8，数据类型必须为float16或float32。

    **输出：**

    Tensor，具有与 `x` 相同的数据类型和shape。

    **异常：**

    - **TypeError** - `x` 的数据类型既不是float16也不是float32。
