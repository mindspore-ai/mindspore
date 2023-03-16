mindspore.ops.Mod
==================

.. py:class:: mindspore.ops.Mod

    将第一个输入Tensor逐元素除以第二个输入Tensor，并取余。

    `x` 和 `y` 的输入遵循隐式类型转换规则，使数据类型一致。输入必须是两个Tensor或一个Tensor和一个Scalar。当输入是两个Tensor时，两个数据类型都不能是bool，它们的shape可以广播。当输入是一个Tensor和一个Scalar时，Scalar只能是一个常量。

    .. math::
        out_{i} = x_{i} \text{ % } y_{i}

    .. warning::
        - 输入数据不支持0。
        - 当输出包含的元素个数超过2048时，该算子不能保证双千分之一的精度要求。
        - 由于架构的差异，在NPU和CPU上生成的结果可能不一致。
        - 如果shape表示为 :math:`(D1, D2, ..., Dn)` ，则 :math:`D1*D2... *DN<=1000000,n<=8` 。

    输入：
        - **x** (Union[Tensor, numbers.Number, bool]) - 第一个输入是数值型、bool，或数据类型为数值型的Tensor。
        - **y** (Union[Tensor, numbers.Number, bool]) - 当第一个输入是Tensor时，第二个输入可以是数值型、bool，或数据类型为数值型的Tensor。当第一个输入是数值型或bool时，第二个输入必须是数据类型为数值型的Tensor。

    输出：
        Tensor，shape与广播后的shape相同，数据类型为两个输入中精度较高或数据类型相对最高的类型。

    异常：
        - **TypeError** - 如果 `x` 和 `y` 都不是以下之一：Tensor、数值型、bool。
        - **TypeError** - 如果 `x` 和 `y` 都不是Tensor。
        - **ValueError** - 如果 `x` 和 `y` 的shape不能相互广播。
