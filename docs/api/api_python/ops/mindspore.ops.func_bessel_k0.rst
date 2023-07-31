mindspore.ops.bessel_k0
=======================

.. py:function:: mindspore.ops.bessel_k0(x)

    逐元素计算第二类零阶修正Bessel函数值。

    计算公式定义如下：

    .. math::
        \begin{array}{ll} \\
            K_{0}(x)= \lim_{\nu \to 0} \left(\frac{\pi}{2}\right) \frac
            {I_{-\nu}(x)-I_{\nu}(x)}{\sin (\nu \pi)} = \int_{0}^{\infty} e^{-x \cosh t} d t
        \end{array}
    
    其中 :math:`I_{0}` 是第一类零阶修正Bessel函数。

    参数：
        - **x** (Tensor) - 输入Tensor。数据类型应为float16，float32或float64。

    返回：
        Tensor，shape和数据类型与 `x` 相同。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `x` 的数据类型不是float16，float32或float64。
