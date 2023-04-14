mindspore.ops.ScatterNd
=======================

.. py:class:: mindspore.ops.ScatterNd

    根据指定的索引将更新值散布到新Tensor上。

    更多参考详见 :func:`mindspore.ops.scatter_nd`。

    输入：
        - **indices** (Tensor) - 指定新Tensor中散布的索引，数据类型为int32或int64。索引的秩须至少为2，并且 `indices_shape[-1] <= len(shape)` 。
        - **updates** (Tensor) - 指定更新Tensor，shape为 `indices_shape[:-1] + shape[indices_shape[-1]:]` 。
        - **shape** (tuple[int]) - 指定输出Tensor的shape，数据类型与索引相同。 `shape` 为 :math:`(x_1, x_2, ..., x_R)` 且 `shape` 的长度大于或等于2。换句话说， `shape` 至少是 :math:`(x_1, x_2)` ，且 `shape` 中的任何元素的值都必须大于等于1。也就是说， :math:`x_1` >= 1， :math:`x_2` >= 1。

    输出：
        Tensor，更新后的Tensor，数据类型与输入 `update` 相同，shape与输入 `shape` 相同。
