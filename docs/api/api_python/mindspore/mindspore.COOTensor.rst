mindspore.COOTensor
===================

.. py:class:: mindspore.COOTensor(indices=None, values=None, shape=None, coo_tensor=None)

    用来表示某一张量在给定索引上非零元素的集合，其中索引(indices)指示了每一个非零元素的位置。

    对一个稠密Tensor `dense` 来说，它对应的COOTensor(indices, values, dense_shape)，满足 `dense[indices[i]] = values[i]` 。

    如果 `indices` 是[[0, 1], [1, 2]]， `values` 是[1, 2]， `dense_shape` 是(3, 4)，那么它对应的稠密Tensor如下：

    .. code-block::

        [[0, 1, 0, 0],
         [0, 0, 2, 0],
         [0, 0, 0, 0]]

    .. note::
        这是一个实验特性，在未来可能会发生API的变化。

    **参数：**

    - **indices** (Tensor) - 形状为 `[N, ndims]` 的二维整数张量，其中N和ndims分别表示稀疏张量中 `values` 的数量和COOTensor维度的数量。
    - **values** (Tensor) - 形状为 `[N]` 的一维张量，用来给 `indices` 中的每个元素提供数值。
    - **shape** (tuple(int)) - 形状为ndims的整数元组，用来指定稀疏矩阵的稠密形状。
    - **coo_tensor** (COOTensor) - COOTensor对象，用来初始化新的COOTensor。

    **返回：**

    COOTensor，由 `indices` 、 `values` 和 `shape` 组成。

    .. py:method:: abs()

        对所有非零元素取绝对值，并返回新的COOTensor。

        **返回：**

        CSRTensor。

    .. py:method:: astype(dtype)

        返回指定数据类型的COOTensor。

        **参数：**

        - **dtype** (`mindspore.dtype`) - 指定数据类型。

        **返回：**

        COOTensor。

    .. py:method:: dtype
        :property:

        返回稀疏矩阵非零元素值数据类型。

    .. py:method:: indices
        :property:

        返回COOTensor的索引值。

    .. py:method:: itemsize
        :property:

        返回每个非零元素所占字节数。

    .. py:method:: ndim
        :property:

        返回稀疏矩阵的稠密维度。

    .. py:method:: shape
        :property:

        返回稀疏矩阵的稠密形状。

    .. py:method:: size
        :property:

        返回稀疏矩阵非零元素值数量。

    .. py:method:: to_csr()

        将COOTensor转换为CSRTensor。

        **返回：**

        CSRTensor。

    .. py:method:: to_dense()

        将COOTensor转换为稠密Tensor。

        **返回：**

        Tensor。

    .. py:method:: to_tuple()

        将COOTensor的索引，非零元素，以及形状信息作为tuple返回。

        **返回：**

        tuple(Tensor, Tensor, tuple(int))

    .. py:method:: values
        :property:

        返回COOTensor的非零元素值。
