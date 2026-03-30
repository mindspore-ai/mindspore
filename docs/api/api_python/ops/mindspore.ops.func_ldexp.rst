mindspore.ops.ldexp
====================

.. py:function:: mindspore.ops.ldexp(x, other)

    逐元素将输入tensor乘以 :math:`2^{other}` 。

    该函数通常用于将尾数 `x` 和由指数 `other` 创建的2的整数次幂相乘来构造浮点数:

    .. warning::
        2.9.0（不含）之后版本中，参数 `x` 将重命名为 `input`。

    .. math::
        out_{i} = x_{i} * ( 2 ^{other_{i}} )

    参数：
        - **x**  (Tensor) - 尾数tensor。
        - **other** (Tensor) - 指数tensor，通常为整数。

    返回：
        Tensor
