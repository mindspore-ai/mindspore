mindspore.ops.TriuIndices
==========================

.. py:class:: mindspore.ops.TriuIndices(row, col, offset=0, dtype=mstype.int32)

    返回一个包含 `row` * `col` 的矩阵的上三角形部分的索引的Tensor。Tensor的shape为 :math:`(2, tril\_size)` ，其中， `tril_size` 为上三角矩阵的元素总数。第一行包含所有索引的行坐标，第二行包含列坐标。索引按行排序，然后按列排序。

    矩阵的下三角形部分定义为对角线本身和对角线以上的元素。

    .. note::
        在CUDA上运行的时候， `row` * `col` 必须小于2^59以防止计算时溢出。

    参数：
        - **row** (int) - 2-D 矩阵的行数。
        - **col** (int) - 2-D 矩阵的列数。
        - **offset** (int，可选) - 对角线偏移量。默认值：0。
        - **dtype** (:class:`mindspore.dtype`，可选) - 指定输出Tensor数据类型，支持的数据类型为 `mstype.int32` 和 `mstype.int64` ，默认值： `mstype.int32` 。

    输出：
        - **y** (Tensor) - 矩阵的下三角形部分的索引。数据类型由 `dtype` 指定，shape为 :math:`(2, tril\_size)` ，其中， `tril_size` 为上三角矩阵的元素总数。

    异常：
        - **TypeError** - 如果 `row` 、 `col` 或 `offset` 不是int。
        - **TypeError** - 如果 `dtype` 的类型不是int32或int64。
        - **ValueError** - 如果 `row` 或者 `col` 小于零。
