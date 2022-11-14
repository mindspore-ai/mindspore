mindspore.ops.masked_fill
=========================

.. py:function:: mindspore.ops.masked_fill(input_x, mask, value)

    将掩码位置为True的位置填充指定的值。`input_x` 和 `mask` 的shape需相同或可广播。

    参数：
        - **input_x** (Tensor) - 输入Tensor，其数据类型为bool, uint8, int8, int16, int32,
          int64, float16, float32, float64, complex64或complex128。
        - **mask** (Tensor[bool]) - 输入的掩码，其数据类型为bool。
        - **value** (Union[float, Tensor]) - 用来填充的值，其数据类型与 `input_x` 相同。

    返回：
        Tensor，输出与输入的数据类型和shape相同。

    异常：
        - **TypeError** - `mask` 的数据类型不是bool。
        - **TypeError** - `input_x` 或 `mask` 不是Tensor。
        - **ValueError** - `input_x` 和 `mask` 的shape不可广播。
        - **TypeError** - `input_x` 或 `value` 的数据类型不是bool, uint8, int8, int16, int32, int64, float16, float32, float64, complex64或complex128。
        - **TypeError** - `value` 的数据类型与 `input_x` 不同。
        - **TypeError** - `value` 既不是float也不是Tensor。



