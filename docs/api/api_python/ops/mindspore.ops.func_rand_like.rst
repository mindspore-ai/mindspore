mindspore.ops.rand_like
========================

.. py:function:: mindspore.ops.rand_like(input, seed=None, *, dtype=None)

    返回一个与输入shape相同的tensor，其元素为服从均匀分布的 :math:`[0, 1)` 区间的随机数。

    .. warning::
        Ascend后端不支持随机数重现功能， `seed` 参数不起作用。
        2.9.0（不含）之后版本非兼容性变更：`seed` 将移除，随机数将改为由 `generator` 控制。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **seed** (int，可选) - 随机种子，必须大于或等于0。默认 ``None`` 。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`，可选) - 指定数据类型。默认 ``None`` 。

    返回：
        Tensor
