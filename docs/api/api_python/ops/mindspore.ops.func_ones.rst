mindspore.ops.ones
===================

.. py:function:: mindspore.ops.ones(shape, dtype=None)

    创建一个值全为1的tensor。

    .. warning::
        2.9.0（不含）之后版本非兼容性变更：`shape` 将改为 `size`，且 `dtype` 将改为关键字参数。当前版本中，
        `shape` 的 Tensor 类型输入也会逐步移除。

    参数：
        - **shape** (Union[tuple[int], list[int], int, Tensor]) - 指定的shape。
        - **dtype** (:class:`mindspore.dtype`) - 指定数据类型，默认 ``None`` 。

    返回：
        Tensor
