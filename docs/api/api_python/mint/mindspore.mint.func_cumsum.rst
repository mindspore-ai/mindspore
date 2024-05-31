mindspore.mint.cumsum
======================

.. py:function:: mindspore.mint.cumsum(input, dim, dtype=None)

    计算输入Tensor `input` 沿轴 `dim` 的累积和。

    .. math::
        y_i = x_1 + x_2 + x_3 + ... + x_i

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **dim** (int) - 累积和计算的轴。
        - **dtype** (:class:`mindspore.dtype`, 可选) - 输出数据类型。如果不为None，则输入会转化为 `dtype`。这有利于防止数值溢出。如果为None，则输出和输入的数据类型一致。默认值： ``None`` 。

    返回：
        Tensor，和输入Tensor的shape相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **ValueError** - 如果 `dim` 超出范围。
