mindspore.ops.ldexp
====================

.. py:function:: mindspore.ops.ldexp(x, other)

    将输入乘以 :math:`2^{other}` 。

    .. math::
        out_{i} = x_{i} * ( 2_{i} ^{other} )

    .. note::
        通常，该函数可以通过将输入中的尾数乘以 `other` 中的指数的整数2的幂来创建浮点数。

    参数：
        - **x**  (Tensor) - 输入的一个Tensor。
        - **other** (Tensor) - 指数Tensor，通常为整数。

    返回：
        Tensor，返回计算的结果。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。
        - **TypeError** - 如果 `other` 不是Tensor。
        - **ValueError** - 如果 `x` 和 `other` 的shape不能广播。
