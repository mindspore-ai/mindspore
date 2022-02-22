mindspore.ops.Maximum
======================

.. py:class:: mindspore.ops.Maximum

    计算输入Tensor的最大值。

    .. note::
        - 输入 `x` 和 `y` 遵循隐式类型转换规则，使数据类型保持一致。
        - 输入必须是两个Tensor，或一个Tensor和一个Scalar。
        - 当输入是两个Tensor时，它们的数据类型不能同时是bool，并保证其shape可以广播。
        - 当输入是一个Tensor和一个Scalar时，Scalar只能是一个常数。
        - 支持广播。
        - 如果一个元素和NaN比较，则返回该元素。

    .. math::
        output_i = max(x_i, y_i)

    **输入：**

    - **x** (Union[Tensor, Number, bool]) - 第一个输入可以是Number或bool，也可以是数据类型为Number或bool的Tensor。
    - **y** (Union[Tensor, Number, bool]) - 第二个输入是Number，当第一个输入是Tensor时，也可以是bool，或数据类型为Number或bool的Tensor。

    **输出：**

    Tensor的shape与广播后的shape相同，数据类型为两个输入中精度较高或数字较多的类型。

    **异常：**

    - **TypeError** - `x`和 `y` 不是以下之一：Tensor，Number，bool。
    - **ValueError** - `x` 和 `y` 的shape不相同。