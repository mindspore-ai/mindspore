mindspore.ops.fill
==================

.. py:function:: mindspore.ops.fill(type, shape, value)

    创建tensor，用指定值填充。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。请使用 :func:`mindspore.ops.full` 代替。

    参数：
        - **type** (mindspore.dtype) - 指定数据类型。
        - **shape** (Union(Tensor, tuple[int])) - 指定的shape。
        - **value** (Union(Tensor, number.Number, bool)) - 指定值。

    返回：
        Tensor
