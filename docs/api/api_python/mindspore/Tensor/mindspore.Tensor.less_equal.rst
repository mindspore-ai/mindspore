mindspore.Tensor.less_equal
===========================

.. py:method:: mindspore.Tensor.less_equal(other)

    逐元素计算 :math:`input <= other` 的bool值。

    .. math::
        out_{i} =\begin{cases}
            & \text{True,    if } input_{i}<=other_{i} \\
            & \text{False,   if } input_{i}>other_{i}
            \end{cases}

    参数：
        - **other** (Union[Tensor, number.Number, bool]) - 第二个输入，当第一个输入是Tensor时，第二个输入应该是一个number.Number或bool值，或数据类型为number或bool_的Tensor。当第一个输入是Scalar时，第二>个输>入必须是数据类型为number或bool_的Tensor。

    返回：
        Tensor，shape与广播后的shape相同，数据类型为bool。

    异常：
        - **TypeError** - `other` 都不是Tensor。
