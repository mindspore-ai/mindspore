mindspore.ops.logical_and
==========================

.. py:function:: mindspore.ops.logical_and(x, y)

    逐元素计算两个Tensor的逻辑与运算。
    `x` 和 `y` 的输入遵循隐式类型转换规则，使数据类型一致。
    输入必须是两个Tensor或一个Tensor和一个bool。
    当输入是两个Tensor时，它们的shape可以广播，数据类型必须都是bool。
    当输入是一个Tensor和一个bool时，bool对象只能是一个常量，Tensor的数据类型必须是bool。

    .. math::
        out_{i} = x_{i} \wedge y_{i}

    .. note::
        logical_and支持广播。

    参数：
        - **x** (Union[Tensor, bool]) - 第一个输入是bool或数据类型可被隐式转换为bool的Tensor。
        - **y** (Union[Tensor, bool]) - 当第一个输入是Tensor的时候，第二个输入是bool或者数据类型可被隐式转换为bool的Tensor。


    返回：
        Tensor，形状与广播后的shape相同，数据类型为bool。

    异常：
        - **TypeError** - 如果 `x` 和 `y` 都不是Tensor。
