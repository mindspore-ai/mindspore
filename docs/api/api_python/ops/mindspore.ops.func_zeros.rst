mindspore.ops.zeros
====================

.. py:function:: mindspore.ops.zeros(size, dtype=None)

    创建一个值全为0的tensor。

    .. warning::
        - 参数 `size` 在2.9.0（不含）之后版本中将不再支持Tensor类型的输入。
        - 2.9.0（不含）之后版本非兼容性变更：`dtype` 将改为关键字参数。

    参数：
        - **size** (Union[tuple[int], list[int], int, Tensor]) - 指定的shape。
        - **dtype** (:class:`mindspore.dtype`, 可选) - 指定数据类型，默认 ``None`` 。

    返回：
        Tensor
