mindspore.ops.bessel_j1
=======================

.. py:function:: mindspore.ops.bessel_j1(x)

    逐元素计算输入tensor的第一类一阶贝塞尔函数值。

    .. warning::
        2.9.0（不含）之后版本中，参数 `x` 将重命名为 `input`。

    .. math::
        \begin{array}{ll} \\
            J_{1}(x) = \frac{1}{\pi} \int_{0}^{\pi} \cos (x \sin \theta- \theta) d \theta
            =\sum_{m=0}^{\infty} \frac{(-1)^{m} x^{2 m+1}}{2^{2 m+1} m !(m+1) !}
        \end{array}

    参数：
        - **x** (Tensor) - 输入tensor。

    返回：
        Tensor
