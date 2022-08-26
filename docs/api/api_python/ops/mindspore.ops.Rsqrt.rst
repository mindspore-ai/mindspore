mindspore.ops.Rsqrt
====================

.. py:class:: mindspore.ops.Rsqrt

    逐元素计算输入Tensor平方根的倒数。

    .. math::
        out_{i} =  \frac{1}{\sqrt{x_{i}}}

    输入：
        - **x** (Tensor) - Rsqrt的输入，其rank必须在[0, 7]（含）中，并且每个元素必须是非负数。

    输出：
        Tensor，数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。
