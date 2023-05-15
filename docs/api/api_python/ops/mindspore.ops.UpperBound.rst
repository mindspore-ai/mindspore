mindspore.ops.UpperBound
=========================

.. py:class:: mindspore.ops.UpperBound(out_type=mstype.int32)

    返回一个Tensor，该Tensor包含用于查找 `values` 的值的在升序排列的 `sorted_x` 中上界的索引。

    参数：
        - **out_type** (:class:`mindspore.dtype`, 可选) - 指定输出数据类型，支持 ``mindspore.dtype.int32`` 和 ``mindspore.dtype.int64`` 。默认值： ``mindspore.dtype.int32`` 。

    输入：
        - **sorted_x** (Tensor) - 数据类型为实数的输入Tensor，其秩必须为2， `sorted_x` 每一行都需要按升序排序。
        - **values** (Tensor) - 数据类型与 `sorted_x` 相同的输入Tensor，其秩必须为2，两个输入的shape[0]必须一致。

    输出：
        Tensor，数据列选由 `out_type` 决定，shape与 `values` 一致。

    异常：
        - **TypeError** - `sorted_x` 不是Tensor。
        - **TypeError** - `values` 不是Tensor。
        - **TypeError** - `sorted_x` 与 `values` 数据类型不一致。
        - **ValueError** - `sorted_x` 的秩不是2。
        - **ValueError** - `values` 的秩不是2。
        - **ValueError** - `sorted_x` 与 `values` 的行数不一致。
