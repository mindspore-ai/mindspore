mindspore.mint.floor
=====================

.. py:function:: mindspore.mint.floor(input)

    逐元素向下取整函数。

    .. math::
        out_i = \lfloor input_i \rfloor

    参数：
        - **input** (Tensor) - 输入Tensor。支持的数据类型: float16、float32、float64、bfloat16、int8、int16、int32、int64、uint8、uint16、uint32、uint64。

    返回：
        Tensor，shape与 `input` 相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `input` 的数据类型不支持。
