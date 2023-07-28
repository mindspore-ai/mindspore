mindspore.ops.bessel_k1e
========================

.. py:function:: mindspore.ops.bessel_k1e(x)

    逐元素计算指数缩放第二类一阶修正Bessel函数值。

    计算公式定义如下：

    .. math::
        \begin{array}{ll} \\
            K_{1}e(x)= e^{(-|x|)} * K_{1}(x) = e^{(-|x|)} * \int_{0}
            ^{\infty} e^{-x \cosh t} \cosh (t) d t
        \end{array}

    其中 :math:`K_{1}` 是第二类一阶修正Bessel函数。

    参数：
        - **x** (Tensor) - Tensor的输入。数据类型应为float16，float32或float64。

    返回：
        Tensor，shape和数据类型与 `x` 相同。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `x` 的数据类型不是float16，float32或float64。
