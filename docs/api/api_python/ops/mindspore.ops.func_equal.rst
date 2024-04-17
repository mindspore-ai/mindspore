mindspore.ops.equal
====================

.. py:function:: mindspore.ops.equal(input, other)

    逐元素比较两个输入Tensor是否相等。

    第二个输入可以是一个shape可以广播成第一个输入的Number或Tensor， 反之亦然。

    .. math::
        out_{i} =\begin{cases}
            & \text{True,    if } input_{i} = other_{i} \\
            & \text{False,   if } input_{i} \ne other_{i}
            \end{cases}

    .. note::
        - `input` 和 `other` 遵循隐式类型转换规则，使数据类型保持一致。
        - 两个输入的shape支持广播。

    参数：
        - **input** (Union[Tensor, Number]) - 第一个输入可以是数值型，也可以是数据类型为数值型的Tensor。
        - **other** (Union[Tensor, Number]) - 第二个输入可以是数值型，也可以是数据类型为数值型的Tensor。

    返回：
        Tensor，输出的shape与输入广播后的shape相同，数据类型为bool。

    异常：
        - **TypeError** - `input` 和 `other` 不是以下之一：Tensor、Number类型。