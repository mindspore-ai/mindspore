mindspore.ops.randn_like
=========================

.. py:function:: mindspore.ops.randn_like(input, seed=None, *, dtype=None)

    返回一个与输入shape相同的tensor，其元素为服从标准正态分布的随机数。

    .. warning::
        Ascend后端不支持随机数重现功能， `seed` 参数不起作用。
        2.9.0（不含）之后版本非兼容性变更：`seed` 将移除，并新增 `generator/device` 相关参数。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **seed** (int，可选) - 随机种子，必须大于或等于0。默认 ``None`` 。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`，可选) - 指定数据类型。默认 ``None`` 。

    返回：
        Tensor
