mindspore.ops.Sqrt
===================

.. py:class:: mindspore.ops.Sqrt

    计算输入Tensor的平方根。

    .. math::
        out_{i} =  \sqrt{x_{i}}

    **输入：**

    - **x** (Tensor) - Sqrt的输入，任意维度的Tensor，其秩应小于8，数据类型为Number。

    **输出：**

    Tensor，shape和数据类型与输入 `x` 相同。

    **异常：**

    - **TypeError** - `x` 不是Tensor。