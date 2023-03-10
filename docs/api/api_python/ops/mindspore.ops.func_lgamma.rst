mindspore.ops.lgamma
=====================

.. py:function:: mindspore.ops.lgamma(input)

    计算输入的绝对值的gamma函数的自然对数。

    .. math::
        \text{out}_{i} = \ln \Gamma(|\text{input}_{i}|)

    参数：
        - **input** (Tensor) - 输入Tensor，数据类型是float16，float32或float64。

    返回：
        Tensor，数据类型和 `input` 一样。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `input` 的数据类型不是float16，float32或float64。
