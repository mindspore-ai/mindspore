mindspore.ops.MaskedFill
=========================

.. py:class:: mindspore.ops.MaskedFill

    将掩码位置为True的位置填充指定的值。

    .. note::
        如果 `value` 是Python类型的浮点数，则默认会转为float32类型。这种情况下，当 `input_x` 为float16类型时，在CPU和Ascend平台上，`input_x` 会转为float32类型参与计算，
        并将结果类型转换到float16类型，可能会造成一定性能损耗；而在GPU平台上，则会引起TypeError。因此建议 `value` 采用与 `input_x` 具有相同数据类型的Tensor。

    更多参考详见 :func:`mindspore.ops.masked_fill`。

    输入：
        - **input** (Tensor) - 输入Tensor。
        - **mask** (Tensor[bool]) - 输入的掩码，其数据类型为bool。
        - **value** (Union[float, Tensor]) - 用来填充的值，只支持零维Tensor或float。

    输出：
        Tensor，输出与输入的数据类型和shape相同。
