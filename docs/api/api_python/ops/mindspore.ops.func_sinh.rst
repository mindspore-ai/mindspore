mindspore.ops.sinh
===================

.. py:function:: mindspore.ops.sinh(input)

    逐元素计算输入Tensor的双曲正弦。

    .. math::
        output_i = \sinh(input_i)

    参数：
        - **input** (Tensor) - sinh的输入Tensor。支持数据类型：

          - GPU/CPU： float16、float32、float64、complex64或complex128。
          - Ascend： bool、int8、uint8、int16、int32、int64、float16、float32、float64、complex64、complex128或bfloat16。

    返回：
        Tensor，shape与 `input` 相同。
        当输入类型为bool、int8、uint8、int16、int32、int64时，返回值类型为float32。
        否则，返回值类型与输入类型相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - 

          - CPU/GPU: 如果 `input` 的数据类型不是float16、float32、float64、complex64或complex128。
          - Ascend: 如果 `input` 的数据类型不是bool、int8、uint8、int16、int32、int64、float16、float32、float64、complex64、complex128或bfloat16。
