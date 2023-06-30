mindspore.ops.SparseGatherV2
============================

.. py:class:: mindspore.ops.SparseGatherV2

    基于指定的索引和axis返回输入Tensor的切片。

    输入：
        - **input_params** (Tensor) - 被切片的Tensor。shape： :math:`(x_1, x_2, ..., x_R)` 。
        - **input_indices** (Tensor)- shape： :math:`(y_1, y_2, ..., y_S)` 。
          指定切片的索引，取值须在 `[0, input_params.shape[axis])` 范围内。
        - **axis** (Union(int, Tensor[int])) - 进行索引的axis。axis是Tensor的时候，size必须是1。

    输出：
        Tensor，shape： :math:`(z_1, z_2, ..., z_N)` 。
