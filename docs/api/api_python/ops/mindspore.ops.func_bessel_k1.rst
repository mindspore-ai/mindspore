mindspore.ops.bessel_k1
=======================

.. py:function:: mindspore.ops.bessel_k1(x)

    逐元素计算输入tensor的第二类一阶修正贝塞尔函数值。

    计算公式定义如下：

    .. math::
        \begin{array}{ll} \\
            K_{1}(x)=\lim_{\nu \to 1} \left(\frac{\pi}{2}\right) \frac{I_{-\nu}(x)-
            I_{\nu}(x)}{\sin (\nu \pi)} = \int_{0}^{\infty} e^{-x \cosh t} \cosh (t) d t
        \end{array}

    参数：
        - **x** (Tensor) - 输入tensor。

    返回：
        Tensor