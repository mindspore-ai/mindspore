mindspore.ops.NotEqual
========================

.. py:class:: mindspore.ops.NotEqual

    计算两个Tensor是否不相等。

    .. note::
        - 输入 `x` 和 `y` 遵循隐式类型转换规则，使数据类型保持一致。
        - 输入必须是两个Tensor，或一个Tensor和一个Scalar。
        - 当输入是两个Tensor时，它们的shape可以广播。
        - 当输入是一个Tensor和一个Scalar时，Scalar只能是一个常数。
        - 支持广播。

    .. math::
        out_{i} =\begin{cases}
            & \text{True,    if } x_{i} \ne y_{i} \\
            & \text{False,   if } x_{i} = y_{i}
            \end{cases}

    **输入：**

    - **x** (Union[Tensor, Number, bool]) - 第一个输入可以是Number或bool，也可以是数据类型为Number或bool的Tensor。
    - **y** (Union[Tensor, Number, bool]) - 第二个输入可以是Number或bool。也可以是数据类型为Number或bool的Tensor。

    **输出：**

    Tensor，输出shape与输入相同，数据类型为bool。

    **异常：**

    - **TypeError** - `x` 和 `y` 不是以下之一：Tensor、Number、bool。
    - **TypeError** - `x` 和 `y` 都不是Tensor。
