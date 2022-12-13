mindspore.ops.greater_equal
===========================

.. py:function:: mindspore.ops.greater_equal(input, other)

    按元素比较输入参数 :math:`input >= other` 的值，输出结果为bool值。

    参数：
        - **input** (Union[Tensor, number.Number, bool]) - 第一个输入，是一个number.Number、bool值或数据类型为 `number <https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 或 `bool_ <https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 的Tensor。
        - **other** (Union[Tensor, number.Number, bool]) - 第二个输入，当第一个输入是Tensor时，第二个输入应该是一个number.Number或bool值，或数据类型为number或bool_的Tensor。当第一个输入是Scalar时，第二>个输入必须是数据类型为number或bool_的Tensor。

    返回：
        Tensor，shape与广播后的shape相同，数据类型为bool。
