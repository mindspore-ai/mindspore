mindspore.ops.copysign
=======================

.. py:function:: mindspore.ops.copysign(x, other)

    逐元素地创建一个新的浮点Tensor，其大小为 `x`，符号为 `other` 的符号。

    参数：
        - **x** (Union[Tensor]) - 要更改符号的值。
        - **other** (Union[int, float, Tensor]) - `other` 的符号被复制到 `x`。如果 `x.shape != other.shape`，`other` 必须可广播为 `x` 的shape(这也是输出的shape)。

    返回：
        Tensor。数据类型为float。`x` 的值加上 `other` 的符号，shape与 `x` 相同。

    异常：
        - **TypeError** - 如果输入的数据类型不在给定的类型中，或者输入不能转换为Tensor。