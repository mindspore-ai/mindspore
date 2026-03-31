mindspore.ops.index_fill
========================

.. py:function:: mindspore.ops.index_fill(x, axis, index, value)

    按照指定轴和索引用输入 `value` 填充输入 `x` 的元素。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    参数：
        - **x** (Tensor) - 输入tensor。
        - **axis** (Union[int, Tensor]) - 指定轴。
        - **index** (Tensor) - 指定索引。
        - **value** (Union[bool, int, float, Tensor]) - 填充输入tensor的值。

    返回：
        Tensor
