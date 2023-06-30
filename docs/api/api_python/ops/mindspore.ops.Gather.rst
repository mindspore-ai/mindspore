mindspore.ops.Gather
======================

.. py:class:: mindspore.ops.Gather(batch_dims=0)

    返回输入Tensor在指定 `axis` 上 `input_indices` 索引对应的元素组成的切片。

    更多参考详见 :func:`mindspore.ops.gather`。

    参数：
        - **batch_dims** (int，可选) - 指定batch维的数量。它必须要小于或等于 `input_indices` 的rank。默认值： ``0`` 。

    输入：
        - **input_params** (Tensor) - 原始Tensor，shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **input_indices** (Tensor) - 要切片的索引Tensor，shape为 :math:`(y_1, y_2, ..., y_S)` 。指定原始Tensor中要切片的索引。数据类型必须是int32或int64。
        - **axis** (Union(int, Tensor[int])) - 指定要切片的维度索引。axis是Tensor的时候，size必须是1。

    输出：
        Tensor，shape为 :math:`input\_params.shape[:axis] + input\_indices.shape + input\_params.shape[axis + 1:]` 。
