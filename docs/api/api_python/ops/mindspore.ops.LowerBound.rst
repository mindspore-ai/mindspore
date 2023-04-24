mindspore.ops.LowerBound
========================

.. py:class:: mindspore.ops.LowerBound(out_type=mstype.int32)

    逐元素查找 `values` 在有序数列 `sorted_x` 中下界的索引。

    参数：
        - **out_type** (:class:`mindspore.dtype`，可选) - 可选的数据类型， ``mindspore.dtype.int32`` 或 ``mindspore.dtype.int64`` 。默认值： ``mindspore.dtype.int32`` 。

    输入：
        - **sorted_x** (Tensor) - 输入Tensor的数据类型为实数且每行数据必须按升序排列。秩必须为2。
        - **values** (Tensor) - 输入Tensor的数据类型与 `sorted_x` 一致，且 `values` 与 `sorted_x` 的第一维必须相等。秩必须为2。
 
    输出：
        Tensor，其数据类型由 `out_type` 决定，其shape与 `values` 相同。

    异常：
        - **TypeError** - 若 `sorted_x` 不是Tensor。
        - **TypeError** - 若 `values` 不是Tensor。
        - **TypeError** - 若 `out_type` 不合法。
        - **TypeError** - 若 `sorted_x` 与 `values` 的类型不一致。
        - **ValueError** - 若 `sorted_x` 的秩不等于2。
        - **ValueError** - 若 `values` 的秩不等于2。
        - **ValueError** - 若 `sorted_x` 和 `values` shape的第一维不相等。
