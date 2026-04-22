mindspore.ops.bessel_i1
=======================

.. py:function:: mindspore.ops.bessel_i1(x)

    逐元素计算输入tensor的第一类一阶修正贝塞尔函数值。

    .. warning::
        2.9.0（不含）之后版本中，参数 `x` 将重命名为 `input`。

    .. math::
        \begin{array}{ll} \\
            I_{1}(x)=\mathrm{i}^{-1} J_{1}(\mathrm{i} x)=\sum_{m=0}^
            {\infty} \frac{x^{2m+1}}{2^{2m+1} m ! (m+1) !}
        \end{array}

    参数：
        - **x** (Tensor) - 输入tensor。

    返回：
        Tensor
