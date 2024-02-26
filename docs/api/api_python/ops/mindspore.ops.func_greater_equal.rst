mindspore.ops.greater_equal
===========================

.. py:function:: mindspore.ops.greater_equal(input, other)

    输入两个Tensor，逐元素比较第一个Tensor是否大于等于第二个Tensor。

    更多参考详见 :func:`mindspore.ops.ge`。

    参数：
        - **input** (Union[Tensor, Number]) - 第一个输入，是一个Number或数据类型为 `number <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 或 `bool_ <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 的Tensor。
        - **other** (Union[Tensor, Number]) - 第二个输入，当第一个输入是Tensor时，第二个输入应该是一个Number或数据类型为number或bool_的Tensor。当第一个输入是Scalar时，第二个输入必须是数据类型为number或bool_的Tensor。

    返回：
        Tensor，shape与广播后的shape相同，数据类型为bool。
