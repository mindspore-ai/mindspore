mindspore.ops.BesselI1
======================

.. py:class:: mindspore.ops.BesselI1

    逐元素计算第一类一阶修正Bessel函数值。

    计算公式定义如下：

    .. math::
        \begin{array}{ll} \\
            I_{1}(x)=\mathrm{i}^{-1} J_{1}(\mathrm{i} x)=\sum_{m=0}^
            {\infty} \frac{x^{2m+1}}{2^{2m+1} m ! (m+1) !}
        \end{array}

    其中 :math:`J_{1}` 是第一类一阶的Bessel函数。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    更多细节请参考 :func:`mindspore.ops.bessel_i1` 。

    输入：
        - **x** (Tensor) - 输入Tensor。数据类型应为float16、float32或float64。

    输出：
        Tensor，shape和数据类型与 `x` 相同。
