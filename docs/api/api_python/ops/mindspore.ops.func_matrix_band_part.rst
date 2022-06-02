mindspore.ops.matrix_band_part
==============================

.. py:function:: mindspore.ops.matrix_band_part(x, lower, upper)

    将每个最内层矩阵的中心带外的所有位置设置为零。


    **参数：**

    - **x** (Tensor) - `x` 的shape为 :math:`(*, m, n)` ，其中 :math:`*` 表示任意batch维度。`x` 的数据类型必须为float16、float32、float64、int32或int64。
    - **lower** (int) - 要保留的子对角线数。`lower` 的数据类型必须是int32或int64。如果为负数，则保留整个下三角形。
    - **upper** (int) - 要保留的子对角线数。`upper` 的数据类型必须是int32或int64。如果为负数，则保留整个上三角形。

    **返回：**

    Tensor，其数据类型和维度必须和输入中的 `x` 保持一致。

    **异常：**

    - **TypeError** - 输入的 `x` 的数据类型不是float16、float32、float64、int32或int64。
    - **TypeError** - 输入的 `lower` 的数据类型不是int32或int64。
    - **TypeError** - 输入的 `upper` 的数据类型不是int32或int64。
    - **ValueError** - `x` 的shape不是大于或等于2维。
