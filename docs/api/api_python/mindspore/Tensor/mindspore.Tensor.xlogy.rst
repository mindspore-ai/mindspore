mindspore.Tensor.xlogy
======================

.. py:method:: mindspore.Tensor.xlogy(y)

    计算原Tensor乘以输入Tensor的对数。当原Tensor为零时，则返回零。原Tensor的数据类型需要是
    `number <https://www.mindspore.cn/docs/zh-CN/r1.9/api_python/mindspore.html#mindspore.dtype>`_ 或
    `bool_ <https://www.mindspore.cn/docs/zh-CN/r1.9/api_python/mindspore.html#mindspore.dtype>`_。
    后面为了使表达清晰，使用 `x` 代替原Tensor。

    .. math::
        out_i = x_{i}\ln{y_{i}}

    `x` 和 `y` 的输入遵循隐式类型转换规则，使数据类型一致。输入必须是两个Tensor或一个Tensor和一个Scalar。当输入是两个Tensor时，它们的数据类型不能同时是bool的，它们的shape可以广播。当输入是一个Tensor和一个Scalar时，Scalar只能是一个常量。

    .. warning::
        - 在Ascend上， `x` 和 `y` 必须为float16或float32。

    参数：
        - **y** (Union[Tensor, number.Number, bool]) - 第二个输入为数值型。当第一个输入是Tensor或数据类型为数值型或bool的Tensor时，则第二个输入是数值型或bool。当第一个输入是Scalar时，则第二个输入必须是数据类型为数值型或bool的Tensor。

    返回：
        Tensor，shape与广播后的shape相同，数据类型为两个输入中精度较高或数数值较高的类型。

    异常：
        - **TypeError** - 如果 `x` 和 `y` 不是数值型、bool或Tensor。
        - **TypeError** - 如果 `x` 和 `y` 的数据类型不是float16、float32、float64、complex64或complex128。
        - **ValueError** - 如果 `x` 不能广播到与 `y` 的shape一致。