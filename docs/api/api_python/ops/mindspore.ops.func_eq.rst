mindspore.ops.eq
================

.. py:function:: mindspore.ops.eq(input, other)

    逐元素比较两个输入Tensor是否相等。

    第二个输入可以是一个shape可以广播成第一个输入的Number或Tensor， 反之亦然。

    .. math::
        out_{i} =\begin{cases}
            & \text{True,    if } input_{i} = other_{i} \\
            & \text{False,   if } input_{i} \ne other_{i}
            \end{cases}

    .. note::
        - `input` 和 `other` 遵循隐式类型转换规则，使数据类型保持一致。
        - 输入必须是两个Tensor，或一个Tensor和一个Scalar。
        - 两个输入的shape支持广播。

    参数：
        - **input** (Union[Tensor, Number]) - 第一个输入可以是数值型，也可以是数据类型为数值型的Tensor。
        - **other** (Union[Tensor, Number]) - 当第一个输入是Tensor时，第二个输入是数值型或数据类型为数值型的Tensor，数据类型与第一个输入相同。当第一个输入是数值型时，第二个输入应为Tensor。

    返回：
        Tensor，输出的shape与输入广播后的shape相同，数据类型为bool。

    异常：
        - **TypeError** - `input` 和 `other` 都不是Tensor。