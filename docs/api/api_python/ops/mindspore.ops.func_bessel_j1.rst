mindspore.ops.bessel_j1
=======================

.. py:function:: mindspore.ops.bessel_j1(x)

    逐元素计算输入数据的第一类一阶的Bessel函数。

    计算公式定义如下：

    .. math::
        \begin{array}{ll} \\
            J_{1}(x) = \frac{1}{\pi} \int_{0}^{\pi} \cos (x \sin \theta- \theta) d \theta
            =\sum_{m=0}^{\infty} \frac{(-1)^{m} x^{2 m+1}}{2^{2 m+1} m !(m+1) !}
        \end{array}

    参数：
        - **x** (Tensor) - Tensor的输入。数据类型应为float16，float32或float64。

    返回：
        Tensor，shape和数据类型与 `x` 相同。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `x` 的数据类型不是float16，float32或float64。
