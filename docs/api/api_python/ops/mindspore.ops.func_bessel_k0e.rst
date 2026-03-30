mindspore.ops.bessel_k0e
========================

.. py:function:: mindspore.ops.bessel_k0e(x)

    逐元素计算输入tensor的指数缩放第二类零阶修正贝塞尔函数值。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    .. math::
        \begin{array}{ll} \\
            K_{0}e(x)= e^{(-|x|)} * K_{0}(x) = e^{(-|x|)} * \int_{0}^
            {\infty} e^{-x \cosh t} d t
        \end{array}

    参数：
        - **x** (Tensor) - 输入tensor。

    返回：
        Tensor
