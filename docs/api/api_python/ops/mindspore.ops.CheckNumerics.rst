mindspore.ops.CheckNumerics
===========================

.. py:class:: mindspore.ops.CheckNumerics

    检查Tensor的NaN和Inf值。如果输入包含NaN或Inf值，则引发运行时错误。

    输入：
        - **x** (Tensor) - 任意维度的Tensor。数据类型为float16、float32或float64。

    输出：
        Tensor，如果 `x` 中没有NaN或Inf值，则具有与 `x` 相同的shape和数据类型。

    异常：
        - **TypeError** - `x` 数据类型不是float16、float32或float64。
        - **RuntimeError** - `x` 的中存在NaN或者Inf值。
