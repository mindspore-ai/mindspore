mindspore.Tensor.masked_fill
============================

.. py:method:: mindspore.Tensor.masked_fill(mask, value)

    将掩码位置为True的位置填充指定的值。该Tensor和 `mask` 的shape需相同或可广播。

    参数：
        - **mask** (Tensor[bool]) - mask矩阵，值为bool类型的Tensor。
        - **value** (Union[float, Tensor]) - 填充值，其数据类型与该Tensor相同。

    返回：
        Tensor，shape和dtype与该Tensor相同。

    异常：
        - **TypeError** - `mask` 不是Tensor。
        - **TypeError** - `mask` 的数据类型不是bool。
        - **ValueError** - 该Tensor和 `mask` 的shape不可广播。
        - **TypeError** - 该Tensor 或 `value` 的数据类型不是float16、float32、int8、或int32。
        - **TypeError** - `value` 的数据类型与该Tensor不同。
        - **TypeError** - `value` 既不是float也不是Tensor。