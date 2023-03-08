mindspore.ops.xlogy
====================

.. py:function:: mindspore.ops.xlogy(input, other)

    计算第一个输入Tensor乘以第二个输入Tensor的对数。当 `input` 为零时，则返回零。

    .. math::
        out_i = input_{i}\ln{other_{i}}

    `input` 和 `other` 的输入遵循隐式类型转换规则，使数据类型一致。输入必须是两个Tensor或一个Tensor和一个Scalar。当输入是两个Tensor时，它们的shape可以广播。当输入是一个Tensor和一个Scalar时，Scalar只能是一个常量。

    .. warning::
        - 在Ascend上， `input` 和 `other` 必须为float16或float32。

    参数：
        - **input** (Union[Tensor, number.Number, bool]) - 第一个输入为数值型。数据类型为 `number <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 或 `bool_ <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 。
        - **other** (Union[Tensor, number.Number, bool]) - 第二个输入为数值型。当第一个输入是Tensor或数据类型为数值型或bool的Tensor时，则第二个输入是数值型或bool。当第一个输入是Scalar时，则第二个输入必须是数据类型为数值型或bool的Tensor。

    返回：
        Tensor，shape与广播后的shape相同，数据类型为两个输入中精度较高或数数值较高的类型。

    异常：
        - **TypeError** - 如果 `input` 和 `other` 不是数值型、bool或Tensor。
        - **TypeError** - 如果 `input` 和 `other` 的数据类型不是float16、float32、float64、complex64或complex128。
        - **ValueError** - 如果 `input` 不能广播到与 `other` 的shape一致。
