mindspore.ops.logaddexp2
========================

.. py:function:: mindspore.ops.logaddexp2(input, other)

    计算以2为底的输入的指数和的对数。

    .. math::

        out_i = log_2(2^{input_i} + 2^{other_i})

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **other** (Tensor) - 输入Tensor。如果 `input` 的shape不等于 `other` 的shape，它们必须被广播成相同shape(输出的形状)。

    返回：
        Tensor。

    异常：
        - **TypeError** - `input` 或 `other` 不是Tensor。
