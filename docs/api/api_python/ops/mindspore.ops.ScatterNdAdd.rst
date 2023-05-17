mindspore.ops.ScatterNdAdd
===========================

.. py:class:: mindspore.ops.ScatterNdAdd(use_locking=False)

    使用给定值通过加法运算和输入索引更新Tensor值。在更新完成后输出 `input_x` 。这有利于更加方便地使用更新后的值。

    更多参考详见 :func:`mindspore.ops.scatter_nd_add`。

    参数：
        - **use_locking** (bool，可选) - 是否启用锁保护。默认值： ``False`` 。

    输入：
        - **input_x** (Parameter) - 输入参数，数据类型是Parameter。其shape为 :math:`(N, *)` ，其中 :math:`*` 为任意数量的额外维度。
        - **indices** (Tensor) - 指定加法操作的索引，数据类型为mindspore.int32。索引的rank必须至少为2，并且 `indices.shape[-1] <= len(shape)` 。
        - **updates** (Tensor) - 指定与 `input_x` 相加操作的Tensor，数据类型与 `input_x` 相同，shape为 `indices.shape[:-1] + x.shape[indices.shape[-1]:]` 。

    输出：
        Tensor，更新后的 `input_x` ，shape和数据类型与 `input_x` 相同。
