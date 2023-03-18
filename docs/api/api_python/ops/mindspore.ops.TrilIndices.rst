mindspore.ops.TrilIndices
==========================

.. py:class:: mindspore.ops.TrilIndices(row, col, offset=0, dtype=mstype.int32)

    计算 `row` * `col` 行列矩阵的下三角元素的索引，并将它们作为一个 2xN 的Tensor返回。

    更多参考详见 :func:`mindspore.ops.tril_indices`。

    参数：
        - **row** (int) - 2-D 矩阵的行数。
        - **col** (int) - 2-D 矩阵的列数。
        - **offset** (int, 可选) - 对角线偏移量。默认值：0。
        - **dtype** (:class:`mindspore.dtype`, 可选) - 指定输出Tensor数据类型，支持的数据类型为 `mstype.int32` 和 `mstype.int64` ，默认值： `mstype.int32` 。

    输出：
        - **y** (Tensor) - 矩阵的下三角形部分的索引。数据类型由 `dtype` 指定，shape为 :math:`(2, tril\_size)` ，其中， `tril_size` 为下三角矩阵的元素总数。
