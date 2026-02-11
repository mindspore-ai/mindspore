mindspore.mint.cumsum
======================

.. py:function:: mindspore.mint.cumsum(input, dim, dtype=None)

    返回tensor在指定维度上累积的元素和。

    .. math::
        y_i = x_1 + x_2 + x_3 + ... + x_i

    参数：
        - **input** (Tensor) - 输入tensor。
        - **dim** (int) - 指定计算维度。
        - **dtype** (:class:`mindspore.dtype`, 可选) - 指定数据类型。默认 ``None`` 。

    返回：
        Tensor
