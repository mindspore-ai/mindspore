mindspore.ops.count_nonzero
============================

.. py:function:: mindspore.ops.count_nonzero(x, axis=(), keep_dims=False, dtype=mindspore.int32)

    计算输入Tensor指定轴上的非零元素的数量。

    参数：
        - **x** (Tensor) - 输入数据用于统计非零元素。 :math:`(N,*)` ，其中 :math:`*` 表示任意数量的附加维度。
        - **axis** (Union[int, tuple(int), list(int)]) - 指定计算的维度。只允许为常量。默认值：()，在所有维度进行计算。
        - **keep_dims** (bool) - 如果为True，则保留计算的维度，且长度为1。如果为False，则不要保留这些维度。默认值：False。
        - **dtype** (Union[Number, mindspore.bool\_]) - 输出Tensor的数据类型。只允许为常量。默认值：mindspore.int32。

    返回：
        Tensor，非零元素的数量。数据类型由 `dtype` 所指定。
