mindspore.ops.DivNoNan
=============================

.. py:class:: mindspore.ops.DivNoNan

    对 `x1` 和 `x2` 逐元素执行安全除法，如果 `x2` 的元素为0，则返回0。

    `x1` 和 `x2` 的输入遵循隐式类型转换规则，使数据类型一致。
    输入必须是两个Tensor或一个Tensor和一个Scalar。
    当输入是两个Tensor时，它们的dtype不能同时是布尔型的，它们的shape可以广播。
    当输入是一个Tensor和一个Scalar时，Scalar只能是一个常数。

    .. math::
        output_{i} = \begin{cases}
        0, & \text{ if } x2_{i} = 0\\
        x1_{i} / x2_{i}, & \text{ if } x2_{i} \ne 0
        \end{cases}

    输入：
        - **x1** (Union[Tensor, number.Number, bool]) - 第一个输入是number.Number、bool或者Tensor，数据类型为 `number <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.html#mindspore.dtype>`_ 或 `bool_ <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.html#mindspore.dtype>`_ 。
        - **x2** (Union[Tensor, number.Number, bool]) - 当第一个输入是bool或数据类型为number或bool\_的Tensor时，第二个输入是number.Number或bool。当第一个输入是Scalar时，第二个输入必须是数据类型为number或bool\_的Tensor。

    输出：
        Tensor，shape与广播后的shape相同，数据类型为两个输入中精度较高或数位较高的类型。

    异常：
        - **TypeError** - 如果 `x1` 和 `x2` 不是number.Number、bool或Tensor。
