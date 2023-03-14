mindspore.ops.expm1
====================

.. py:function:: mindspore.ops.expm1(input)

    逐元素计算输入Tensor的指数，然后减去1。

    .. math::
        out_i = e^{x_i} - 1

    参数：
        - **input** (Tensor) - 数据类型为float16或float32的Tensor，其秩必须在[0, 7]中。

    返回：
        Tensor，shape与 `input` 相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - 如果 `input` 的数据类型既不是float16也不是float32。
