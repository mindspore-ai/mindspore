mindspore.ops.MaskedFill
=========================

.. py:class:: mindspore.ops.MaskedFill()

    将掩码位置为True的位置填充指定的值。

    `input` 和 `mask` 的shape需相同或可广播。

    **输入：**

    - **input** (Tensor) - 输入Tensor，其数据类型为float16、float32、int8、或int32。
    - **mask** (Tensor[bool]) - 输入的掩码，其数据类型为bool。
    - **value** (Union[float, Tensor]) - 用来填充的值，只支持0维Tensor或float。

    **输出：**

    Tensor，输出与输入的数据类型和shape相同。

    **异常：**

    - **TypeError** - `input` 或 `mask` 不是Tensor。
    - **TypeError** - `value` 既不是float也不是Tensor。
    - **TypeError** - `input` 或 `value` 的数据类型不是float16、float32、int8、或int32。
    - **TypeError** - `value` 的数据类型与 `input` 不同。
    - **TypeError** - `mask` 的数据类型不是bool。
    - **ValueError** - `input` 和 `mask` 的shape不可广播。