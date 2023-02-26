mindspore.ops.deg2rad
======================

.. py:function:: mindspore.ops.deg2rad(x)

    逐元素地将 `x` 从度数制转换为弧度制。

    参数：
        - **x** (Tensor[Number]) - 输入的Tensor。其必须是一个正定矩阵，数据类型为float16，float32或float64。

    返回：
        Tensor，其数据类型与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是一个Tensor。
        - **TypeError** - 如果 `x` 的数据类型不是float16，float32或float64。
