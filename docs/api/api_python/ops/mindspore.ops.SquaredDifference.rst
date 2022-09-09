mindspore.ops.SquaredDifference
================================

.. py:class:: mindspore.ops.SquaredDifference

    第一个输入Tensor元素中减去第二个输入Tensor，并返回其平方。

    `x` 和 `y` 的输入遵循隐式类型转换规则，使数据类型一致。输入必须是两个Tensor或一个Tensor和一个Scalar。当输入是两个Tensor时，它们的数据类型不能同时为bool类型，并且它们的shape可以广播。当输入是一个Tensor和一个Scalar时，Scalar只能是一个常量。

    .. math::
        out_{i} = (x_{i} - y_{i}) * (x_{i} - y_{i}) = (x_{i} - y_{i})^2

    输入：
        - **x** (Union[Tensor, Number, bool]) - 第一个输入，为数值型，或为bool，或数据类型为float16、float32、int32或bool的Tensor。
        - **y** (Union[Tensor, Number, bool]) - 第二个输入，通常为数值型，如果第一个输入是数据类型为float16、float32、int32或bool的Tensor时，第二个输入是bool。

    输出：
        Tensor，shape与广播后的shape相同，数据类型为两个输入中精度较高或数字较高的类型。

    异常：
        - **TypeError** - 如果 `x` 和 `y` 不是数值型、bool或Tensor。
