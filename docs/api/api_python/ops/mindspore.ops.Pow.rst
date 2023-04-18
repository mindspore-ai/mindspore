mindspore.ops.Pow
==================

.. py:class:: mindspore.ops.Pow

    计算 `x` 中每个元素的 `y` 次幂。

    更多参考详见 :func:`mindspore.ops.pow`。

    输入：
        - **x** (Union[Tensor, number.Number, bool]) - 第一个输入，是一个number.Number、bool值或数据类型为 `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ 或 `bool_ <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ 的Tensor。
        - **y** (Union[Tensor, number.Number, bool]) - 第二个输入，当第一个输入是Tensor时，第二个输入应该是一个number.Number或bool值，或数据类型为number或bool_的Tensor。当第一个输入是Scalar时，第二个输入必须是数据类型为number或bool_的Tensor。

    输出：
        Tensor，shape与广播后的shape相同，数据类型为两个输入中精度较高的类型。
