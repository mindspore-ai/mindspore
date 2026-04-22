mindspore.ops.bessel_i0
=======================

.. py:function:: mindspore.ops.bessel_i0(x)

    逐元素计算输入tensor的第一类零阶修正贝塞尔函数值。

    .. warning::
        2.9.0（不含）之后版本中，参数 `x` 将重命名为 `input`。

    .. math::
        \begin{array}{ll} \\
            I_{0}(x)=J_{0}(\mathrm{i} x)=\sum_{m=0}^{\infty}
            \frac{x^{2 m}}{2^{2 m} (m !)^{2}}
        \end{array}

    参数：
        - **x** (Tensor) - 输入tensor。

    返回：
        Tensor
