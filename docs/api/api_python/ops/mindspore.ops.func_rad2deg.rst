mindspore.ops.rad2deg
======================

.. py:function:: mindspore.ops.rad2deg(x)

    计算一个新的Tensor，其中 `x` 的每个角度元素都从弧度转换为度。

    参数：
        - **x** (Tensor) - 输入的Tensor。

    返回：
        Tensor，其数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是一个Tensor。
        - **TypeError** - 如果 `x` 的数据类型不是float16，float32或float64。
