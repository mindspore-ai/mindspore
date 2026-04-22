mindspore.ops.count_nonzero
============================

.. py:function:: mindspore.ops.count_nonzero(x, axis=(), keep_dims=False, dtype=mstype.int32)

    计算输入tensor指定轴上的非零元素的数量。如果没有指定轴，则计算tensor中所有非零元素的数量。

    .. warning::
        从2.9.0（不含）之后版本开始，参数 `x` 将重命名为 `input` ，
        参数 `axis` 将重命名为 `dim` ，并且 `keep_dims` 和 `dtype` 将不再支持。

    参数：
        - **x** (Tensor) - 输入tensor。
        - **axis** (Union[int, tuple(int), list(int)]，可选) - 指定轴。默认 ``()`` ，计算所有非零元素的个数。
        - **keep_dims** (bool, 可选) - 是否保留 `axis` 指定的维度。默认 ``False`` ，不保留对应维度。
        - **dtype** (Union[Number, mindspore.bool]，可选) - 指定数据类型。默认 ``mstype.int32`` 。

    返回：
        Tensor
