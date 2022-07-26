mindspore.ops.MulNoNan
=======================

.. py:class:: mindspore.ops.MulNoNan

    逐元素计算输入乘积。如果 `y` 为零，无论 `x` 取何值，它都将返回0。

    `x` 和 `y` 的输入遵循隐式类型转换规则，使数据类型一致。输入必须是两个Tensor或一个Tensor和一个Scalar。当输入是两个Tensor时，它们的shape可以被广播。当输入是一个Tensor和一个Scalar时，Scalar只能是一个常量。

    .. math::
        output_{ij} = \begin{cases}
        0, & y_{ij} = 0;\\
        x_{ij} * y_{ij}, & otherwise.
        \end{cases}

    .. note::
        `x` 和 `y` 的shape应该相同，或者可以广播。如果 `y` 是NaN或为无限的，并且 `x` 是0，结果将是NaN。

    输入：
        - **x** (Union[Tensor]) - 第一个输入是Tensor，其数据类型为int32、int64、float16、float32、float64、complex64、complex128或Scalar。
        - **y** (Union[Tensor]) - 第二个输入是Tensor，其数据类型为int32、int64、float16、float32、float64、complex64、complex128或Scalar。

    输出：
        Tensor，shape与广播后的shape相同，数据类型是两个输入中精度较高的类型。

    异常：
        - **TypeError** - 如果 `x` 和 `y` 都不是Tensor。

