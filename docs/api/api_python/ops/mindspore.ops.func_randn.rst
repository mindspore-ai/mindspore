mindspore.ops.randn
====================

.. py:function:: mindspore.ops.randn(*size, dtype=None, seed=None)

    基于输入的 `size` 和 `dtype` ，返回一个tensor，其元素为服从标准正态分布的随机数字。

    .. warning::
        Ascend后端不支持随机数重现功能， `seed` 参数不起作用。

    参数：
        - **size** (Union[int, tuple(int), list(int)]) - 输出tensor的shape。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`，可选) - 指定数据类型。默认 ``None`` 。
        - **seed** (int，可选) - 随机种子，必须大于或等于0。默认 ``None`` 。

    返回：
        Tensor
