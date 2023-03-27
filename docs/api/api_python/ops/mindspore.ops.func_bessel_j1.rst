mindspore.ops.bessel_j1
=======================

.. py:function:: mindspore.ops.bessel_j1(x)

    逐元素计算并返回输入Tensor的Bessel j1函数值。

    参数：
        - **x** (Tensor) - Tensor的输入。数据类型应为float16，float32或float64。

    返回：
        Tensor，shape和数据类型与 `x` 相同。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `x` 的数据类型不是float16，float32或float64。
