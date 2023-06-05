mindspore.ops.expm1
====================

.. py:function:: mindspore.ops.expm1(input)

    逐元素计算输入Tensor的指数，然后减去1。

    .. math::
        out_i = e^{x_i} - 1

    参数：
        - **input** (Tensor) - 输入Tensor。

    返回：
        Tensor，shape与 `input` 相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
