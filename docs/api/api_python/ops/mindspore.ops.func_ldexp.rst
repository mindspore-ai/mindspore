mindspore.ops.ldexp
====================

.. py:function:: mindspore.ops.ldexp(x, other)

    逐元素将输入Tensor乘以 :math:`2^{other}` 。

    该函数使用两个参数，即尾数 `x` 和指数 `other` ，并将它们的乘积作为浮点数返回：

    .. math::
        out_{i} = x_{i} * ( 2_{i} ^{other} )

    .. note::
        该函数通常用于由尾数和幂构造浮点数，或将浮点数按二的幂进行缩放。

    参数：
        - **x**  (Tensor) - 输入的一个Tensor。
        - **other** (Tensor) - 指数Tensor，通常为整数。

    返回：
        Tensor，返回计算的结果。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。
        - **TypeError** - 如果 `other` 不是Tensor。
        - **ValueError** - 如果 `x` 和 `other` 的shape不能广播。
