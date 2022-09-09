mindspore.ops.scatter_nd
========================

.. py:function:: mindspore.ops.scatter_nd(indices, updates, shape)

    根据指定的索引将更新值散布到新Tensor上。

    使用给定的 `shape` 创建一个空Tensor，并将 `updates` 的值通过索引（ `indices` ）来设置空Tensor的值。空Tensor的秩为 :math:`P` ， `indices` 的秩为 :math:`Q` 。

    `shape` 为 :math:`(s_0, s_1, ..., s_{P-1})` ，其中 :math:`P \ge 1` 。

    `indices` 的shape为 :math:`(i_0, i_1, ..., i_{Q-2}, N)` ，其中 :math:`Q \ge 2` 且 :math:`N \le P` 。

    `indices` 的最后一个维度（长度为 :math:`N` ）表示沿着空Tensor的第 :math:`N` 个维度进行切片。

    `updates` 是一个秩为 :math:`Q-1+P-N` 的Tensor，shape为 :math:`(i_0, i_1, ..., i_{Q-2}, s_N, s_{N+1}, ..., s_{P-1})` 。

    如果 `indices` 包含重复的下标，则 `updates` 的值会被累加到同一个位置上。

    在秩为3的第一个维度中插入两个新值矩阵的计算过程如下图所示：

    .. image:: ScatterNd.png

    参数：
        - **indices** (Tensor) - 指定新Tensor中散布的索引，数据类型为int32或int64。索引的秩须至少为2，并且 `indices.shape[-1] <= len(shape)` 。
        - **updates** (Tensor) - 指定更新Tensor，shape为 `indices.shape[:-1] + shape[indices.shape[-1]:]` 。
        - **shape** (tuple[int]) - 指定输出Tensor的shape，数据类型与 `indices` 相同。 `shape` 不能为空，且 `shape` 中的任何元素的值都必须大于等于1。

    返回：
        Tensor，更新后的Tensor，数据类型与输入 `update` 相同，shape与输入 `shape` 相同。

    异常：
        - **TypeError** - `shape` 不是tuple。
        - **ValueError** - `shape` 的任何元素小于1。
