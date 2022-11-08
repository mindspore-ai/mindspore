mindspore.ops.xdivy
====================

.. py:function:: mindspore.ops.xdivy(x, y)

    将第一个输入Tensor除以第二个输入Tensor。当 `x` 为零时，则返回零。

    `x` 和 `y` 的输入遵循隐式类型转换规则，使数据类型一致。输入必须是两个Tensor或一个Tensor和一个Scalar。当输入是两个Tensor时，它们的数据类型不能同时为bool，它们的shape可以广播。当输入是一个Tensor和一个Scalar时，Scalar只能是一个常量。

    .. note::
        当 `x` 和 `y` 数据类型都为复数的时候， 须同时为complex64或者complex128。

    参数：
        - **x** (Union[Tensor, Number, bool]) - number.Number或bool类型的Tensor，或者是一个bool或者number。
        - **y** (Union[Tensor, Number, bool]) - number.Number或bool类型的Tensor，或者是一个bool或者number， `x` 和 `y` 不能同时为bool类型。

    返回：
        Tensor，shape与广播后的shape相同，数据类型为两个输入中精度较高或数据类型较高的类型。

    异常：
        - **TypeError** - 如果 `x` 和 `y` 不是以下之一：Tensor、Number、bool。
        - **TypeError** - 如果 `x` 和 `y` 的数据类型不是float16、float32、float64、complex64、complex128、bool。
        - **ValueError** - 如果 `x` 不能广播至 `y` 的shape。
        - **RuntimeError** - 如果Parameter的 `x` , `y` 需要进行数据类型转换，但是Parameter是不支持数据类型转换。
