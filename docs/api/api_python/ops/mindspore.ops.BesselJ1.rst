mindspore.ops.BesselJ1
======================

.. py:class:: mindspore.ops.BesselJ1

    逐元素计算输入数据的第一类一阶的Bessel函数。

    计算公式定义如下：

    .. math::
        \begin{array}{ll} \\
            J_{1}(x) = \frac{1}{\pi} \int_{0}^{\pi} \cos (x \sin \theta- \theta) d \theta
            =\sum_{m=0}^{\infty} \frac{(-1)^{m} x^{2 m+1}}{2^{2 m+1} m !(m+1) !}
        \end{array}

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    输入：
        - **x** (Tensor) - 输入Tensor。数据类型应为float16、float32或float64。

    输出：
        Tensor，shape和数据类型与 `x` 相同。

    异常：
        - **TypeError** - `x` 不是float16、float32或float64数据类型的Tensor。
