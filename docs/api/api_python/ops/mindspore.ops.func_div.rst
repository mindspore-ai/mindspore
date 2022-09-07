mindspore.ops.div
==================

.. py:function:: mindspore.ops.div(x, y)

    逐元素计算第一个输入Tensor除以第二输入Tensor的商。

    `x` 和 `y` 的输入遵循隐式类型转换规则，使数据类型一致。
    输入必须是两个Tensor，或一个Tensor和一个Scalar。
    当输入是两个Tensor时，它们的数据类型不能同时为bool，它们的shape可以广播。
    当输入是一个Tensor和一个Scalar时，Scalar只能是一个常量。

    .. math::
        out_{i} = x_{i} / y_{i}

    参数：
        - **x** (Union[Tensor, Number, bool]) - 第一个输入，为数值型，或bool，或数据类型为数值型或bool的Tensor。
        - **y** (Union[Tensor, Number, bool]) - 第二个输入，当第一个输入是Tensor时，第二个输入必须是一个数值型或bool，或是数据类型为数值型或bool的Tensor。

    返回：
        Tensor，输出的shape与广播后的shape相同，数据类型取两个输入中精度较高或数字较高的。

    异常：
        - **TypeError** - 如果 `x` 和 `y` 不是以下之一：Tensor、Number、bool。
