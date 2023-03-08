mindspore.ops.floor
====================

.. py:function:: mindspore.ops.floor(input)

    逐元素向下取整函数。

    .. math::
        out_i = \lfloor x_i \rfloor

    参数：
        - **input** (Tensor) - floor的输入，任意维度的Tensor，秩应小于8。其数据类型必须为float16、float32。

    返回：
        Tensor，shape与 `input` 相同。

    异常：
        - **TypeError** - `input` 的数据类型不是Tensor。
        - **TypeError** - `input` 的数据类型不是float16、float32、float64。