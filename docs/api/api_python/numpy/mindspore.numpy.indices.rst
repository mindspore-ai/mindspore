mindspore.numpy.indices
=================================

.. py:function:: mindspore.numpy.indices(dimensions, dtype=mstype.int32, sparse=False)

    返回一个表示网格索引的数组。
    计算一个数组，其中子数组包含仅沿相应轴变化的索引值0，1，...。

    参数：
        - **dimensions** (Union[list(int), tuple]) - 网格的shape。
        - **dtype** (mindspore.dtype, 可选) - 指定结果的数据类型。
        - **sparse** (bool, 可选) - 返回网格的稀疏表示而非密集表示，默认值： ``False`` 。

    返回：
        Tensor，或元素为Tensor的Tuple。如果 ``sparse`` 为 ``False`` ，则返回一个网格索引数组： ``grid.shape = (len(dimensions),) + tuple(dimensions)`` 。如果 ``sparse`` 为 ``True`` ，则返回数组的Tuple，其中 ``grid[i].shape = (1, ..., 1, dimensions[i], 1, ..., 1)`` ， ``dimensions[i]`` 位于第 ``i`` 个位置。

    异常：
        - **TypeError** - 如果输入的 ``dimensions`` 不是Tuple或List。