mindspore.ops.inv
=================

.. py:function:: mindspore.ops.inv(x)

    逐元素计算输入tensor的倒数。

    .. warning::
        2.9.0（不含）之后版本中，参数 `x` 将重命名为 `A`。

    .. math::
        out_i = \frac{1}{x_{i}}

    参数：
        - **x** (Tensor) - 输入tensor。

    返回：
        Tensor
