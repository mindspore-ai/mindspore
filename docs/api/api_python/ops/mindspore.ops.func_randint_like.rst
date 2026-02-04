mindspore.ops.randint_like
===========================

.. py:function:: mindspore.ops.randint_like(input, low, high, seed=None, *, dtype=None)

    返回一个与输入shape相同的tensor，其元素为 [ `low` , `high` ) 区间的随机整数。

    .. warning::
        Ascend后端不支持随机数重现功能， `seed` 参数不起作用。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **low** (int) - 随机区间的起始值。
        - **high** (int) - 随机区间的末尾值。
        - **seed** (int，可选) - 随机种子，必须大于或等于0。默认 ``None`` 。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`，可选) - 指定数据类型。默认 ``None`` 。

    返回：
        Tensor
