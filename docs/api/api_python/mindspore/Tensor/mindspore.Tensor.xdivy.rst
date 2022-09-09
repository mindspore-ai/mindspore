mindspore.Tensor.xdivy
======================

.. py:method:: mindspore.Tensor.xdivy(y)

    计算原Tensor除以输入的Tensor。当原Tensor为零时，则返回零。原Tensor的数据类型需要是float，complex或bool。
    后面为了使表达清晰，使用 `x` 代替原Tensor。

    .. math::
        out_i = x_{i}\y_{i}

    `x` 和 `y` 的输入遵循隐式类型转换规则使数据类型一致。y必须是一个Tensor或Scalar，当y是Tensor时，x和y的数据类型不能同时是bool类型，它们的shape可以广播。当y是Scalar时，只能是一个常量。

    参数：
        - **y** (Union[Tensor, number.Number, bool]) - 当第一个输入x为Tensor的时候，第二个输入y可以是Number类型、bool类型或者数据类型为float16、float32、float64、complex64、complex128、bool的Tensor。

    返回：
        Tensor，shape与广播后的shape相同，数据类型为两个输入中精度较高或数数值较高的类型。

    异常：
        - **TypeError** - 如果 `y` 不是以下之一：Tensor、Number、bool。
        - **TypeError** - 如果 `x` 和 `y` 的数据类型不是float16、float32、float64、complex64、complex128、bool。
        - **ValueError** - 如果 `x` 不能广播至 `y` 的shape。
        - **RuntimeError** - 如果Parameter的 `x` , `y` 需要进行数据类型转换，但是Parameter是不支持数据类型转换。