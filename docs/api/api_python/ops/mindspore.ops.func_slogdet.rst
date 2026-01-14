mindspore.ops.slogdet
=====================

.. py:function:: mindspore.ops.slogdet(input)

    对一个或多个方阵行列式的绝对值取对数，返回其符号和值。

    .. note::
        输出的类型是实数，即使 `input` 是复数。

    参数：
        - **input** (Tensor) - 输入tensor，shape为 :math:`(*, M, M)` 。

    返回：
        两个tensor组成的tuple，分别为符号和绝对值的对数值。