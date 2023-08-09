mindspore.ops.bessel_y0
=======================

.. py:function:: mindspore.ops.bessel_y0(x)

    逐元素计算输入数据的第二类零阶Bessel函数。

    计算公式定义如下：

    .. math::
        \begin{array}{ll} \\
            Y_{0}(x)=\lim_{n \to 0} \frac{J_{n}(x) \cos n \pi-J_{-n}(x)}{\sin n \pi}
        \end{array}

    其中 :math:`J_{0}` 第一类零阶的Bessel函数。

    参数：
        - **x** (Tensor) - 任意维度的Tensor。数据类型应为float16，float32或float64。

    返回：
        Tensor，shape和数据类型与 `x` 相同。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `x` 的数据类型不是float16，float32或float64。
