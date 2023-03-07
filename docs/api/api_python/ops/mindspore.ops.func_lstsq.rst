mindspore.ops.lstsq
===================

.. py:function:: mindspore.ops.lstsq(input, A)

    计算满秩矩阵 `x` :math:`(m \times n)` 与满秩矩阵 `a` :math:`(m \times k)` 的最小二乘问题或最小范数问题的解。

    若 :math:`m \geq n` ， `Lstsq` 解决最小二乘问题：

    .. math::
       \begin{array}{ll}
       \min_y & \|xy-a\|_2.
       \end{array}

    若 :math:`m < n` ， `Lstsq` 解决最小范数问题：

    .. math::
       \begin{array}{llll}
       \min_y & \|y\|_2 & \text{subject to} & xy = a.
       \end{array}

    其中 `y` 表示返回的Tensor。

    参数：
        - **input** (Tensor) - 上述公式中 :math:`(m \times n)` 的矩阵 :math:`x` ，输入Tensor的数据类型为float16、float32或float64。
        - **A** (Tensor) - 上述公式中 :math:`(m \times k)` 的矩阵 :math:`a` ，输入Tensor的数据类型为float16、float32或float64。

    返回：
        Tensor，最小二乘问题或最小范数问题的解，其shape为 :math:`(n \times k)` ，数据类型与 `input` 相同。

    异常：
        - **TypeError** - `input` 或 `A` 不是Tensor。
        - **TypeError** - `input` 或 `A` 不是以下数据类型之一：float16、float32、float64。
        - **TypeError** - `input` 和 `A` 的数据类型不同。
        - **ValueError** - `input` 的维度不等于2。
        - **ValueError** - `A` 的维度不等于2或1。
        - **ValueError** - `input` 与 `A` shape的第零维不相等。
