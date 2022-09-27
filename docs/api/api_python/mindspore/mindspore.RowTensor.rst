mindspore.RowTensor
===================

.. py:class:: mindspore.RowTensor(indices=None, values=None, shape=None, row_tensor=None)

    用来表示一组指定索引的张量切片的稀疏表示。

    通常用于表示一个有着形状为[L0, D1, .., DN]的更大的稠密张量（其中L0>>D0）的子集。

    其中，参数 `indices` 用于指定 `RowTensor` 从该稠密张量的第一维度的哪些位置来进行切片。

    由 `RowTensor` 切片表示的稠密张量具有以下属性： `dense[slices.indices[i], :, :, :, ...] = slices.values[i, :, :, :, ...]` 。

    如果 `indices` 是[0]， `values` 是[[1, 2]]， `shape` 是(3, 2)，那么它对应的稠密Tensor如下：

    .. code-block::

        [[1, 2],
         [0, 0],
         [0, 0]]

    .. note::
        这是一个实验特性，在未来可能会发生API的变化。

    参数：
        - **indices** (Tensor) - 形状为[D0]的一维整数张量。默认值：None。
        - **values** (Tensor) - 形状为[D0, D1, ..., Dn]中任意类型的张量。默认值：None。
        - **shape** (tuple(int)) - 包含相应稠密张量形状的整数元组。默认值：None。
        - **row_tensor** (RowTensor) - RowTensor对象，用来初始化新的RowTensor。默认值：None。

    返回：
        RowTensor，由 `indices` 、 `values` 和 `shape` 组成。

    .. py:method:: dense_shape
        :property:

        返回稀疏矩阵的稠密形状。

    .. py:method:: indices
        :property:

        返回RowTensor的索引值。
    
    .. py:method:: values
        :property:

        返回RowTensor的非零元素值。
