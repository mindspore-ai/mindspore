mindspore.SparseTensor
======================

.. py:class:: mindspore.SparseTensor(indices, values, dense_shape)

    用来表示某一张量在给定索引上非零元素的集合。

    `SparseTensor` 只能在  `Cell` 的构造方法中使用。

    .. note::
        目前不支持PyNative模式。

    对于稠密张量，其 `SparseTensor(indices, values, dense_shape)` 具有 `dense[indices[i]] = values[i]` 。

    **参数：**

    - **indices** (Tensor) - 形状为 `[N, ndims]` 的二维整数张量，其中N和ndims分别表示稀疏张量中 `values` 的数量和SparseTensor维度的数量。
    - **values** (Tensor) - 形状为[N]的一维张量，其内部可以为任何数据类型，用来给 `indices` 中的每个元素提供数值。
    - **dense_shape** (tuple(int)) - 形状为ndims的整数元组，用来指定稀疏矩阵的稠密形状。

    **返回：**

    SparseTensor，由 `indices` 、 `values` 和 `dense_shape` 组成。

    **样例：**

    >>> import mindspore as ms
    >>> import mindspore.nn as nn
    >>> from mindspore import SparseTensor
    >>> class Net(nn.Cell)：
    ...     def __init__(self, dense_shape)：
    ...         super(Net, self).__init__()
    ...         self.dense_shape = dense_shape
    ...     def construct(self, indices, values)：
    ...         x = SparseTensor(indices, values, self.dense_shape)
    ...         return x.values, x.indices, x.dense_shape
    >>>
    >>> indices = Tensor([[0, 1], [1, 2]])
    >>> values = Tensor([1, 2], dtype=ms.float32)
    >>> out = Net((3, 4))(indices, values)
    >>> print(out[0])
    [1.2.]
    >>> print(out[1])
    [[0 1]
    [1 2]]
    >>> print(out[2])
    (3, 4)
