mindspore.mint.floor_divide
===========================

.. py:function:: mindspore.mint.floor_divide(input, other) -> Tensor

    按元素将第一个输入tensor除以第二个输入tensor，并向下取整。

    `input` 和 `other` 的输入遵循隐式类型转换规则，使数据类型一致。输入必须是两个tensor，或一个tensor和一个scalar。当输入是两个tensor时，它们的数据类型不能同时为bool，其shape可以广播。当输入是一个tensor和一个scalar时，scalar只能是一个常量。

    .. math::
        out_{i} = \text{floor}( \frac{input_i}{other_i})

    其中 :math:`floor` 表示Floor算子。有关更多详细信息，请参阅 :class:`mindspore.mint.floor` 算子。

    参数：
        - **input** (Union[Tensor, Number, bool]) - 第一个输入，为数值型、bool，或数据类型为数值型或bool的Tensor。
        - **other** (Union[Tensor, Number, bool]) - 第二个输入，为数值型、bool，或数据类型为数值型或bool的Tensor。

    返回：
        Tensor，输出的shape与广播后的shape相同，数据类型取两个输入中精度较高或位数较高的。

    异常：
        - **TypeError** - 如果 `input` 和 `other` 不是以下之一：Tensor、number.Number或bool。
