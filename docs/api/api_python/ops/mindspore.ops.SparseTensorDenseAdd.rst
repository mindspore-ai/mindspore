mindspore.ops.SparseTensorDenseAdd
==================================

.. py:class:: mindspore.ops.SparseTensorDenseAdd

    一个稀疏tensor加上稠密Tensor得到一个稠密Tensor。

    输入：
        - **x1_indices** (Tensor) - 二维Tensor，表示稀疏Tensor的索引位置，shape为 :math:`(n, ndim)` ，
          支持的数据类型为int32和int64，其值必须为非负数。
        - **x1_values** (Tensor) - 一维Tensor，表示稀疏Tensor索引位置对应值，shape为 :math:`(n,)`。
        - **x1_shape** (tuple(int)) - 稀疏Tensor对应的稠密Tensor的shape，是一个不含负数， 长度为ndim的tuple，
          shape为 :math:`(ndim,)`。
        - **x2** (Tensor) - 稠密Tensor，数据类型与稀疏Tensor一致。

    输出：
        Tensor，shape与 `x1_shape` 一致。

    异常：
        - **TypeError** - `x1_indices` 和 `x1_shape` 不是int32或者int64。
        - **ValueError** - `x1_shape` 的shape， `x1_indices` 的shape， `x1_values` 的shape和 `x2` 的shape不满足参数描述。
