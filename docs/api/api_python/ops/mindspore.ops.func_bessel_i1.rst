mindspore.ops.bessel_i1
=======================

.. py:function:: mindspore.ops.bessel_i1(x)

    逐元素计算第一类一阶修正Bessel函数值。

    计算公式定义如下：

    .. math::
        \begin{array}{ll} \\
            I_{1}(x)=\mathrm{i}^{-1} J_{1}(\mathrm{i} x)=\sum_{m=0}^
            {\infty} \frac{x^{2m+1}}{2^{2m+1} m ! (m+1) !}
        \end{array}

    其中 :math:`J_{1}` 是第一类一阶的Bessel函数。

    参数：
        - **x** (Tensor) - 任意维度的Tensor。数据类型应为float16，float32或float64。

    返回：
        Tensor，shape和数据类型与 `x` 相同。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `x` 的数据类型不是float16，float32或float64。
