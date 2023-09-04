mindspore.RowTensor
===================

.. py:class:: mindspore.RowTensor(indices=None, values=None, shape=None, row_tensor=None)

    用来表示一组指定索引的Tensor切片的稀疏表示。

    若RowTensor的 `values` 的shape为 :math:`(d_0, d_1, ..., d_n)`，则该RowTensor用于表示一个有着shape为 :math:`(l_0, d_1, .., d_n)` 的更大的稠密Tensor的子集，
    其中 :math:`d_i` 为RowTensor第i轴的size， :math:`l_0` 为稠密Tensor在第0轴的size，并且 :math:`l_0 > d_0` 。

    其中，参数 `indices` 用于指定 `RowTensor` 从该稠密Tensor的第一维度的哪些位置来进行切片，
    即参数 `indices` 和 `values` 满足以下关系： :math:`dense[indices[i], :, :, :, ...] = values[i, :, :, :, ...]` 。

    如果 `indices` 是[0]， `values` 是[[1, 2]]， `shape` 是 :math:`(3, 2)` ，那么它对应的稠密Tensor如下：

    .. code-block::

        [[1, 2],
         [0, 0],
         [0, 0]]

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **indices** (Tensor) - shape为 :math:`(d_0)` 的一维整数Tensor。默认值： ``None`` 。
        - **values** (Tensor) - shape为 :math:`(d_0, d_1, ..., d_n)` 中任意类型的Tensor。默认值： ``None`` 。
        - **shape** (tuple(int)) - 包含相应稠密Tensor shape的整数元组。默认值： ``None`` 。
        - **row_tensor** (RowTensor) - RowTensor对象，用来初始化新的RowTensor。默认值： ``None`` 。

    返回：
        RowTensor，由 `indices` 、 `values` 和 `shape` 组成。
