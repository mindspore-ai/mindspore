mindspore.ops.mul
=================

.. py:function:: mindspore.ops.mul(input, other)

    两个Tensor逐元素相乘。

    .. math::

        out_{i} = input_{i} * other_{i}

    .. note::
        - 两个输入中至少有一个Tensor，当两个输入具有不同的shape时，它们的shape必须要能广播为一个共同的shape。
        - 两个输入不能同时为bool类型。[True, Tensor(True, bool\_), Tensor(np.array([True]), bool\_)]等都为bool类型。
        - 两个输入遵循隐式类型转换规则，使数据类型保持一致。

    参数：
        - **input** (Union[Tensor, number.Number, bool]) - 第一个输入，是一个number.Number、bool值或数据类型为 `number <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 或 `bool_ <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 的Tensor。
        - **other** (Union[Tensor, number.Number, bool]) - 第二个输入，当第一个输入是Tensor时，第二个输入应该是一个number.Number或bool值，或数据类型为number或bool的Tensor。当第一个输入是Scalar时，第二个输入必须是数据类型为number或bool的Tensor。

    返回：
        Tensor，shape与广播后的shape相同，数据类型为两个输入中精度较高的类型。

    异常：
        - **TypeError** - `input` 和 `other` 不是Tensor、number.Number或bool。
        - **ValueError** - `input` 和 `other` 的shape不相同。
