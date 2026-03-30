mindspore.ops.bessel_y0
=======================

.. py:function:: mindspore.ops.bessel_y0(x)

    逐元素计算输入tensor的第二类零阶贝塞尔函数值。

    .. warning::
        2.9.0（不含）之后版本中，参数 `x` 将重命名为 `input`。

    .. math::
        \begin{array}{ll} \\
            Y_{0}(x)=\lim_{n \to 0} \frac{J_{n}(x) \cos n \pi-J_{-n}(x)}{\sin n \pi}
        \end{array}

    参数：
        - **x** (Tensor) - 输入tensor。

    返回：
        Tensor
