mindspore.ops.bessel_y1
=======================

.. py:function:: mindspore.ops.bessel_y1(x)

    逐元素计算输入数据的第二类一阶Bessel函数。

    计算公式定义如下：

    .. math::
        \begin{array}{ll} \\
            Y_{1}(x)=\lim_{n \to 1} \frac{J_{n}(x) \cos n \pi-J_{-n}(x)}{\sin n \pi}
        \end{array}

    其中 :math:`J_{1}` 第一类一阶的Bessel函数。

    参数：
        - **x** (Tensor) - 输入Tensor。数据类型应为float16，float32或float64。

    返回：
        Tensor，shape和数据类型与 `x` 相同。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `x` 的数据类型不是float16，float32或float64。
