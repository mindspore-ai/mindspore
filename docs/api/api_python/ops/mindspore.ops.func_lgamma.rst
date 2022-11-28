mindspore.ops.lgamma
=====================

.. py:function:: mindspore.ops.lgamma(x)

    计算输入的绝对值的gamma函数的自然对数。

    .. math::
        \text{out}_{i} = \ln \Gamma(|\text{input}_{i}|)

    参数：
        - **x** (Tensor) - 输入Tensor，数据类型是float16，float32或float64。

    返回：
        Tensor，数据类型和 `x` 一样。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `x` 的数据类型不是float16，float32或float64。
