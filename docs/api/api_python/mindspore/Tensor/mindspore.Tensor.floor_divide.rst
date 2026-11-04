mindspore.Tensor.floor_divide
=============================

.. py:method:: mindspore.Tensor.floor_divide(other) -> Tensor

    按元素将 `self` Tensor除以输入Tensor，并向下取整。

    `self` 和 `other` 遵循隐式类型转换规则，以保证两者数据类型一致。输入必须是两个Tensor或一个Tensor和一个Scalar。当输入是两个Tensor时，它们的数据类型不能同时为bool，其shape可以广播。当输入是一个Tensor和一个Scalar时，Scalar只能是一个常量。

    .. math::
        out_{i} = \text{floor}( \frac{self_i}{other_i})

    其中 :math:`floor` 表示Floor算子。有关更多详细信息，请参阅 :class:`mindspore.mint.floor` 算子。

    参数：
        - **other** (Union[Tensor, Number, bool]) - 第二个输入，为数值型、bool，或者数据类型为数值型或bool的Tensor。

    返回：
        Tensor，输出的shape与广播后的shape相同，数据类型取两个输入中精度较高或数字较高的。

    异常：
        - **TypeError** - 如果 `self` 和 `other` 不是以下之一：Tensor、number.Number或bool。
