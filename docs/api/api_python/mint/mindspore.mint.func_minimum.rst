mindspore.mint.minimum
=======================

.. py:function:: mindspore.mint.minimum(input, other)

    逐元素计算两个输入Tensor中的最小值。

    .. note::
        - 输入 `input` 和 `other` 遵循隐式类型转换规则，使数据类型保持一致。
        - 当输入是两个Tensor时，它们的数据类型不能同时是bool。
        - 当输入是一个Tensor和一个Scalar时，Scalar只能是一个常数。
        - 支持广播。
        - 如果一个元素和NaN比较，则返回NaN。

    .. math::
        output_i = \min(input_i, other_i)

    参数：
        - **input** (Union[Tensor, Number, bool]) - 第一个输入可以是Number或bool，也可以是数据类型为Number或bool的Tensor。
        - **other** (Union[Tensor, Number, bool]) - 第二个输入可以是Number或bool，也可以是数据类型为Number或bool的Tensor。

    返回：
        一个Tensor，其shape与广播后的shape相同，其数据类型为两个输入中精度较高的类型。

    异常：
        - **TypeError** - `input` 和 `other` 不是以下之一：Tensor、Number、bool。
        - **ValueError** - `input` 和 `other` 的广播后的shape不相同。
