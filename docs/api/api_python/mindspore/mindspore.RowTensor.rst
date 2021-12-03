mindspore.RowTensor
===================

.. py:class:: mindspore.RowTensor(indices, values, dense_shape)

    用来表示一组指定索引的张量切片的稀疏表示。

    通常用于表示一个有着形状为[L0, D1, .., DN]的更大的稠密张量（其中L0>>D0）的子集。

    其中，参数 `indices` 用于指定 `RowTensor` 从该稠密张量的第一维度的哪些位置来进行切片。

    由 `RowTensor` 切片表示的稠密张量具有以下属性： `dense[slices.indices[i], :, :, :, ...] = slices.values[i, :, :, :, ...]` 。

    `RowTensor` 只能在  `Cell` 的构造方法中使用。

    目前不支持PyNative模式。

    **参数：**

    - **indices** (Tensor) - 形状为[D0]的一维整数张量。
    - **values** (Tensor) - 形状为[D0, D1, ..., Dn]中任意类型的张量。
    - **dense_shape** (tuple(int)) - 包含相应稠密张量形状的整数元组。

    **返回：**

    RowTensor，由 `indices` 、 `values` 和 `dense_shape` 组成。

    **样例：**

    >>> import mindspore as ms
    >>> import mindspore.nn as nn
    >>> from mindspore import RowTensor
    >>> class Net(nn.Cell):
    ...     def __init__(self, dense_shape)：
    ...         super(Net, self).__init__()
    ...         self.dense_shape = dense_shape
    ...     def construct(self, indices, values)：
    ...         x = RowTensor(indices, values, self.dense_shape)
    ...         return x.values, x.indices, x.dense_shape
    >>>
    >>> indices = Tensor([0])
    >>> values = Tensor([[1, 2]], dtype=ms.float32)
    >>> out = Net((3, 2))(indices, values)
    >>> print(out[0])
    [[1.2.]]
    >>> print(out[1])
    [0]
    >>> print(out[2])
    (3, 2)
