mindspore.ops.ne
========================

.. py:function:: mindspore.ops.ne(input, other)

    计算两个Tensor是否不相等。

    .. note::
        - 输入 `input` 和 `other` 遵循隐式类型转换规则，使数据类型保持一致。
        - 输入必须是两个Tensor，或一个Tensor和一个Scalar。
        - 当输入是两个Tensor时，它们的shape可以广播。
        - 当输入是一个Tensor和一个Scalar时，Scalar只能是一个常数。
        - 支持广播。

    .. math::
        out_{i} =\begin{cases}
        & \text{True,    if } input_{i} \ne other_{i} \\
        & \text{False,   if } input_{i} = other_{i}
        \end{cases}

    参数：
        - **input** (Union[Tensor, Number, bool]) - 第一个输入可以是数值型或bool，也可以是数据类型为数值型或bool的Tensor。
        - **other** (Union[Tensor, Number, bool]) - 第二个输入可以是数值型或bool。也可以是数据类型为数值型或bool的Tensor。

    返回：
        Tensor，输出shape与输入相同，数据类型为bool。

    异常：
        - **TypeError** - `input` 和 `other` 不是以下之一：Tensor、数值型、bool。
        - **TypeError** - `input` 和 `other` 都不是Tensor。
