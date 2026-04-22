mindspore.ops.bessel_k0
=======================

.. py:function:: mindspore.ops.bessel_k0(x)

    逐元素计算输入tensor的第二类零阶修正贝塞尔函数值。

    .. math::
        \begin{array}{ll} \\
            K_{0}(x)= \lim_{\nu \to 0} \left(\frac{\pi}{2}\right) \frac
            {I_{-\nu}(x)-I_{\nu}(x)}{\sin (\nu \pi)} = \int_{0}^{\infty} e^{-x \cosh t} d t
        \end{array}

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    参数：
        - **x** (Tensor) - 输入tensor。

    返回：
        Tensor
