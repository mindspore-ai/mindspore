mindspore.ops.tan
===================

.. py:function:: mindspore.ops.tan(input)

    计算输入元素的正切值。

    .. math::
        out_i = tan(input_i)

    参数：
        - **input** (Tensor) - Tan的输入，任意维度的Tensor。

    返回：
        Tensor，数据类型和shape与 `input` 相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
