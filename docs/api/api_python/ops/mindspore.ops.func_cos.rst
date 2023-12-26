mindspore.ops.cos
==================

.. py:function:: mindspore.ops.cos(input)

    逐元素计算输入的余弦。

    .. math::
        out_i = \cos(x_i)

    .. warning::
        如果使用float64，可能会存在精度丢失的问题。

    参数：
        - **input** (Tensor) - Tensor的shape为 :math:`(N,*)` ，其中 :math:`*` 表示任意数量的附加维度。

    返回：
        Tensor，shape与 `input` 相同。
        当输入类型为bool, int8, uint8, int16, int32, int64时，返回值类型为float32。
        否则，返回值类型与输入类型相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - CPU/GPU: 如果 `input` 的数据类型不是float16、float32、float64、complex64或complex128。
                          Ascend: 如果 `input` 的数据类型不是bool, int8, uint8, int16, int32, int64, float16、float32、float64、complex64或complex128。
