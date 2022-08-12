mindspore.Tensor.minimum
========================

.. py:method:: mindspore.Tensor.minimum(other)

    逐元素计算当前Tensor和输入 `other` 中的最小值。

    .. note::
        - 当前Tensor和输入 `other` 遵循隐式类型转换规则，使数据类型保持一致。
        - 输入 `other` 必须是Tensor或Scalar。
        - 当输入 `other` 是Tensor时，当前Tensor和输入 `other` 的数据类型不能同时是bool。
        - 当输入是Scalar时，Scalar只能是一个常数。
        - 支持广播。
        - 如果一个元素和NaN比较，则返回该元素。

    .. math::
        output_i = min(x_i, y_i)

    参数：
        - **other** (Union[Tensor, Number, bool]) - 可以是Number或bool或Tensor。

    返回：
        一个Tensor，其shape与广播后的shape相同，其数据类型为两个输入中精度较高的类型。

    异常：
        - **TypeError** - `other` 不是以下之一：Tensor，Number，bool。
        - **ValueError** - `other` 和当前Tensor的广播后的shape不相同。