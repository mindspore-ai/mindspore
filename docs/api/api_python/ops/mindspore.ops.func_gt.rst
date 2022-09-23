mindspore.ops.gt
=====================

.. py:function:: mindspore.ops.gt(x, y)

    按元素比较输入参数 :math:`x,y` 的值，输出结果为bool值。

    .. math::
        out_{i} =\begin{cases}
            & \text{True,    if } x_{i}>y_{i} \\
            & \text{False,   if } x_{i}<=y_{i}
            \end{cases}

    .. note::
        - 输入 `x` 和 `y` 遵循隐式类型转换规则，使数据类型保持一致。
        - 输入必须是两个Tensor，或一个Tensor和一个Scalar。
        - 当输入是两个Tensor时，它们的数据类型不能同时是bool，并保证其shape可以广播。
        - 当输入是一个Tensor和一个Scalar时，Scalar只能是一个常数。
        - 支持广播。
        - 若输入的Tensor可以广播，则会把低维度通过复制该维度的值的方式扩展到另一个输入中对应的高维度。

    参数：
        - **x** (Union[Tensor, number.Number, bool]) - 第一个输入，是一个number.Number、bool值或数据类型为 `number <https://www.mindspore.cn/docs/zh-CN/r1.9/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 或 `bool_ <https://www.mindspore.cn/docs/zh-CN/r1.9/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 的Tensor。
        - **y** (Union[Tensor, number.Number, bool]) - 第二个输入，当第一个输入是Tensor时，第二个输入应该是一个number.Number或bool值，或数据类型为number或bool_的Tensor。当第一个输入是Scalar时，第二个输入必须是数据类型为number或bool_的Tensor。

    返回：
        Tensor，shape与广播后的shape相同，数据类型为bool。

    异常：
        - **TypeError** - `x` 和 `y` 都不是Tensor。
