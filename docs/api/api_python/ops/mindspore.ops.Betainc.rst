mindspore.ops.Betainc
=====================

.. py:class:: mindspore.ops.Betainc

    计算正则化不完全beta积分 :math:`I_{x}(a, b)`。

    正则化的不完全beta积分定义为：

    .. math::

        I_{x}(a, b)=\frac{B(x ; a, b)}{B(a, b)}

    其中

    .. math::

        B(x ; a, b)=\int_{0}^{x} t^{a-1}(1-t)^{b-1} d t

    是不完全的 beta 函数， :math:`B(a, b)` 是完整的beta函数。

    输入：
        - **a** (Tensor) - float32或者float64类型的Tensor。
        - **b** (Tensor) - 必须具有与 `a` 相同的dtype和shape的Tensor。
        - **x** (Tensor) - 必须具有与 `a` 相同的dtype和shape的Tensor。

    输出：
        Tensor，具有与 `a` 相同的shape和dtype。

    异常：
        - **TypeError** - 如果 `a` 的dtype不是float32或float64。
        - **TypeError** - 如果 `b` 和 `x` 其中之一的dtype不是和 `a` 的dtype一致。
        - **ValueError** - 如果 `b` 和 `x` 其中之一的shape不是和 `a` 的shape一致。
