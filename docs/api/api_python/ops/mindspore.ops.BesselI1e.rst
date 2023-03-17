mindspore.ops.BesselI1e
========================

.. py:class:: mindspore.ops.BesselI1e

    逐元素计算输入数据的BesselI1e函数值。

    计算公式定义如下：

    .. math::
        BesselI1e(x) = \exp(|x|) * bessel\_i1(x)

    其中bessel_i1是第一类1阶的Bessel函数。

    输入：
        - **x** (Tensor) - 输入Tensor。数据类型应为float16，float32或float64。

    输出：
        Tensor，shape和数据类型与 `x` 相同。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `x` 的数据类型不是float16，float32或float64。
