mindspore.ops.SparseTensorDenseAdd
==================================

.. py:class:: mindspore.ops.SparseTensorDenseAdd

    一个稀疏张量加上稠密张量得到一个稠密张量。

    输入：
        - **x1_indices** (Tensor) - 二维张量，表示稀疏张量的索引位置，shape为 :math:`(n, 2)`。
        - **x1_values** (Tensor) - 一维张量，表示稀疏张量索引位置对应值，shape为 :math:`(n,)`。
        - **x1_shape** (tuple(int)) - 稀疏张量对应的稠密张量维度，shape为 :math:`(N, C)`。
        - **x2** (Tensor) - 稠密张量，数据类型与稀疏张量一致。

    输出：
        张量，shape与 `x1_shape` 一致。

    异常：
        - **TypeError** - `x1_indices` 和 `x1_shape` 不是int32或者int64。
        - **ValueError** - `x1_shape` 的shape， `x1_indices` 的shape， `x1_values` 的shape和 `x2` 的shape不满足参数描述。
