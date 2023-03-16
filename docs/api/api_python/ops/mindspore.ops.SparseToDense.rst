mindspore.ops.SparseToDense
============================

.. py:class:: mindspore.ops.SparseToDense

    将稀疏Tensor转换为密集Tensor。

    输入：
        - **indices** (Tensor) - 二维Tensor，表示元素在稀疏Tensor中的位置。支持int32、int64，每个元素值都应该是非负的。shape是 :math:`(n, 2)` 。
        - **values** (Tensor) - 一维Tensor，表示 `indices` 位置上对应的值。shape为 :math:`(n,)` 。
        - **sparse_shape** (tuple(int)) - 指定稀疏Tensor的shape，由两个正整数组成，表示稀疏Tensor的shape为 :math:`(N, C)` 。

    输出：
        Tensor，计算后的Tensor。数据类型与 `values` 相同，shape由 `sparse_shape` 所指定。

    异常：
        - **TypeError** - 如果 `indices` 的数据类型既不是int32也不是int64。
        - **ValueError** - 如果 `sparse_shape` 、 `indices` 和 `values` 的shape不符合参数中所描述支持的数据类型。
