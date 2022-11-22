mindspore.ops.logaddexp2
========================

.. py:function:: mindspore.ops.logaddexp2(x1, x2)

    计算以2为底的输入的指数和的对数。

    .. math::

        out_i = log_2(2^{x1_i} + 2^{x2_i})

    参数：
        - **x1** (Tensor) - 输入Tensor。
        - **x2** (Tensor) - 输入Tensor。如果 `x1` 的shape不等于 `x2` 的shape，它们必须被广播成相同shape(输出的形状)。

    返回：
        Tensor。

    异常：
        - **TypeError** - `x1` 或 `x2` 不是Tensor。
