mindspore.ops.Div
=================

.. py:class:: mindspore.ops.Div()

    逐元素计算第一输入Tensor除以第二输入Tensor的商。

    .. math::

        out_{i} = \frac{x_i}{y_i}

    .. note::
        - 输入 `x` 和 `y` 遵循 `隐式类型转换规则 <https://www.mindspore.cn/docs/zh-CN/r1.9/note/operator_list_implicit.html>`_ ，使数据类型保持一致。
        - 输入必须是两个Tensor，或一个Tensor和一个Scalar。
        - 当输入是两个Tensor时，它们的数据类型不能同时是bool，并保证其shape可以广播。
        - 当输入是一个Tensor和一个Scalar时，Scalar只能是一个常数。

    输入：
        - **x** (Union[Tensor, number.Number, bool]) - 第一个输入，是一个number.Number、bool值或数据类型为 `number <https://www.mindspore.cn/docs/zh-CN/r1.9/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 或 `bool_ <https://www.mindspore.cn/docs/zh-CN/r1.9/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 的Tensor。
        - **y** (Union[Tensor, number.Number, bool]) - 第二个输入，当第一个输入是Tensor时，第二个输入应该是一个number.Number或bool值，或数据类型为number或bool_的Tensor。当第一个输入是Scalar时，第二个输入必须是数据类型为number或bool_的Tensor。

    输出：
        Tensor，shape与输入 `x`，`y` 广播后的shape相同，数据类型为两个输入中精度较高的类型。

    异常：
        - **TypeError** - `x` 和 `y` 都不是Tensor。
        - **TypeError** - `x` 和 `y` 数据类型都是bool_的Tensor。
