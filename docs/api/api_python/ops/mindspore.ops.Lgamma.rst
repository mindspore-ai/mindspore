mindspore.ops.Lgamma
====================

.. py:class:: mindspore.ops.Lgamma

    计算输入 `x` 伽马函数的自然对数。

    .. math::
        \text{out}_{i} = \ln \Gamma(\text{input}_{i})

    输入：
        - **x** (Tensor) - 输入Tensor，数据类型支持float16、float32和float64。

    输出：
        Tensor，与 `x` 的数据类型相同。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - 输入 `x` 的数据类型不是以下之一：float16、float32、float64。
