mindspore.Tensor.empty_strided
===============================

.. py:method:: mindspore.Tensor.empty_strided(size, stride, dtype=mstype.float64, seed=None)

    使用指定的“大小”和“步幅”创建张量，并用未定义的数据填充。

    参数：
        - **size** (tuple：ints) - 输出张量的形状。
        - **stride** (tuple：ints) - 输出张量的步幅。
        - **dtype** (mindspore.dtype，可选) - 返回张量的所需数据类型。

    返回：
        具有指定大小和步幅并填充了未定义数据的张量。

    平台：
        ``Ascend`` ``GPU`` `` CPU``
