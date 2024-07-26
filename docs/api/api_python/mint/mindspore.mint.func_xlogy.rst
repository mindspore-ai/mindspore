mindspore.mint.xlogy
====================

.. py:function:: mindspore.mint.xlogy(input, other)

    计算第一个输入乘以第二个输入的对数。当 `input` 为零时，则返回零。

    .. math::
        out_i = input_{i}\log{other_{i}}

    `input` 和 `other` 的输入遵循隐式类型转换规则，使数据类型一致。输入必须是两个Tensor或一个Tensor和一个Scalar。当输入是两个Tensor时，它们的shape可以广播。


    参数：
        - **input** (Union[Tensor, number.Number, bool]) - 第一个输入为数值型、bool或数据类型为 `number <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 或 `bool_ <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 的Tensor。
        - **other** (Union[Tensor, number.Number, bool]) - 第二个输入为数值型、bool或数据类型为数值型或bool的Tensor。当第一个输入是Tensor时，则第二个输入是数值型、bool或数据类型为数值型或bool的Tensor。当第一个输入是Scalar时，则第二个输入必须是数据类型为数值型或bool的Tensor。

    返回：
        Tensor，shape与广播后的shape相同，数据类型为两个输入中精度较高或数值较高的类型。

    异常：
        - **TypeError** - 如果 `input` 和 `other` 不是数值型、bool或Tensor。
        - **ValueError** - 如果 `input` 不能广播到与 `other` 的shape一致。
