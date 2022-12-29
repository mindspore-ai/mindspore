mindspore.ops.matrix_band_part
==============================

.. py:function:: mindspore.ops.matrix_band_part(x, lower, upper)

    将矩阵的每个中心带外的所有位置设置为0。

    参数：
        - **x** (Tensor) - `x` 的shape为 :math:`(*, m, n)` ，其中 :math:`*` 表示任意batch维度。`x` 的数据类型必须为float16、float32、float64、int32或int64。
        - **lower** (Union[int, Tensor]) - 要保留的下部子对角线数。其数据类型必须是int32或int64。如果为负数，则保留整个下三角形。
        - **upper** (Union[int, Tensor]) - 要保留的上部子对角线数。其数据类型必须是int32或int64。如果为负数，则保留整个上三角形。

    返回：
        Tensor，其数据类型和维度必须和输入中的 `x` 保持一致。

    异常：
        - **TypeError** - `x` 不是一个Tensor。
        - **TypeError** - `x` 的数据类型不是float16、float32、float64、int32或int64。
        - **TypeError** - `lower` 不是一个数值或者Tensor。
        - **TypeError** - `upper` 不是一个数值或者Tensor。
        - **TypeError** - `lower` 的数据类型不是int32或int64。
        - **TypeError** - `upper` 的数据类型不是int32或int64。
        - **ValueError** - `x` 的shape不是大于或等于二维。
        - **ValueError** - `lower` 的shape不等于零维。
        - **ValueError** - `upper` 的shape不等于零维。
