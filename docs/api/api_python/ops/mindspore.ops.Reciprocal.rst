mindspore.ops.Reciprocal
=========================

.. py:class:: mindspore.ops.Reciprocal

    返回输入Tensor的倒数。

    .. math::
        out_{i} =  \frac{1}{x_{i}}

    输入：
        - **x** (Tensor) - 输入Tensor。

    输出：
        Tensor，shape与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。