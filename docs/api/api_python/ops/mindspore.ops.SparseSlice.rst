mindspore.ops.SparseSlice
==========================

.. py:class:: mindspore.ops.SparseSlice

    基于 `start` 和 `size` 对稀疏Tensor进行切片。

    输入：
        - **indices** (Tensor) - 稀疏Tensor的索引，是一个二维Tensor，shape为 :math:`(N, R)` 。支持的数据类型为int64，其每一个值都必须是非负整数。
        - **values** (Tensor)- 稀疏Tensor `indices` 对应位置的值，是一个一维Tensor，shape为 :math:`(N,)` 。
        - **shape** (Tensor) -  稀疏Tensor的shape，是一个一维Tensor，shape为 :math:`(R,)` ，数据类型为int64。
        - **start** (Tensor) - 切片操作起始位置，是一个一维Tensor，shape为 :math:`(R,)` ，数据类型为int64。
        - **size** (Tensor) - 切片的尺寸，是一个一维Tensor，shape为 :math:`(R,)` ，数据类型为int64。

    输出：
        切片操作生成一个 `SparseTensor` 对象。

        - y_indices: int64类型的Tensor。
        - y_values: 数据类型与 `values` 相同的Tensor.
        - y_shape: int64类型的Tensor，其大小与 `size` 相同。

    异常：
        - **TypeError** - `indices`、 `shape`、 `start` 和 `size` 不是int64数据类型。
        - **ValueError** - `indices` 不是2维Tensor。
        - **ValueError** - `values`、 `start`、 `shape` 和 `size` 不是一维Tensor。
        - **ValueError** - `indices` 和 `values` 对应元素数量不一致。
        - **ValueError** - `indices[1]` 与 `shape` 的shape不一致。
        - **ValueError** - `shape` 与 `start` 的shape不一致。
        - **ValueError** - `shape` 与 `size` git的shape不一致。
