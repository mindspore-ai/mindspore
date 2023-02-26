mindspore.ops.rad2deg
======================

.. py:function:: mindspore.ops.rad2deg(x)

    逐元素地将 `x` 从弧度制转换为度数制。

    参数：
        - **x** (Tensor) - 输入的Tensor。

    返回：
        Tensor，其数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是一个Tensor。
        - **TypeError** - 如果 `x` 的数据类型不是float16，float32或float64。
