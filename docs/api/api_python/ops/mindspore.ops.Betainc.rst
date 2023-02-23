mindspore.ops.Betainc
=====================

.. py:class:: mindspore.ops.Betainc

    计算正则化不完全beta函数 :math:`I_{x}(a, b)`。

    正则化的不完全beta函数定义为：

    .. math::

        I_{x}(a, b)=\frac{B(x ; a, b)}{B(a, b)}

    其中

    .. math::

        B(x ; a, b)=\int_{0}^{x} t^{a-1}(1-t)^{b-1} dt

    是不完全 beta 函数，而

    .. math::

        B(a, b) = \int_0^1 t^{a-1} (1-t)^{b-1} dt

    是完全 beta 函数。

    输入：
        - **a** (Tensor) - beta分布峰值位置。float32或者float64类型的Tensor。
        - **b** (Tensor) - beta分布宽窄度。必须具有与 `a` 相同的dtype和shape的Tensor。
        - **x** (Tensor) - 不完全beta函数积分上限。必须具有与 `a` 相同的dtype和shape的Tensor。

    输出：
        Tensor，具有与 `a` 相同的shape和dtype。

    异常：
        - **TypeError** - 如果 `a` 的dtype不是float32或float64。
        - **TypeError** - 如果 `b` 和 `x` 其中之一的dtype不是和 `a` 的dtype一致。
        - **ValueError** - 如果 `b` 和 `x` 其中之一的shape不是和 `a` 的shape一致。
