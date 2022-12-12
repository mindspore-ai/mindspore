mindspore.ops.t
===============

.. py:function:: mindspore.ops.t(x)

    转置2维Tensor。1维Tensor按原样返回。

    参数：
        - **x** (Tensor) - 输入Tensor。

    返回：
        Tensor，`x` 的转置。

    异常：
        - **TypeError** - `x` 的维度大于2。
