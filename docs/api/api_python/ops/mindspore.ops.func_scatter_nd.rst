mindspore.ops.scatter_nd
========================

.. py:function:: mindspore.ops.scatter_nd(indices, updates, shape)

    根据指定的索引将更新值散布到新tensor上。

    使用给定的 `shape` 创建一个空tensor，并将 `updates` 的值通过索引（ `indices` ）来设置空tensor的值。空tensor的秩为 :math:`P` ， `indices` 的秩为 :math:`Q` 。

    `shape` 为 :math:`(s_0, s_1, ..., s_{P-1})` ，其中 :math:`P \ge 1` 。

    `indices` 的shape为 :math:`(i_0, i_1, ..., i_{Q-2}, N)` ，其中 :math:`Q \ge 2` 且 :math:`N \le P` 。

    `indices` 的最后一个维度（长度为 :math:`N` ）表示沿着空tensor的第 :math:`N` 个维度进行切片。

    `updates` 是一个秩为 :math:`Q-1+P-N` 的tensor，shape为 :math:`(i_0, i_1, ..., i_{Q-2}, s_N, s_{N+1}, ..., s_{P-1})` 。

    如果 `indices` 包含重复的下标，则 `updates` 的值会被累加到同一个位置上。

    在秩为3的第一个维度中插入两个新值矩阵的计算过程如下图所示：

    .. image:: ScatterNd.png

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    参数：
        - **indices** (Tensor) - 指定新tensor中散布的索引。索引的秩须至少为2，并且 `indices.shape[-1] <= len(shape)` 。
        - **updates** (Tensor) - 指定更新tensor。shape为 `indices.shape[:-1] + shape[indices.shape[-1]:]` 。
        - **shape** (tuple[int]) - 指定输出tensor的shape。 `shape` 不能为空，且 `shape` 中的任何元素的值都必须大于等于1。

    返回：
        Tensor
