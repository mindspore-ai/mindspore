mindspore.mint.remainder
========================

.. py:function:: mindspore.mint.remainder(input, other)

    计算第一个输入除以第二个输入后的余数。结果与除数同号且绝对值小于除数的绝对值。

    支持广播和隐式数据类型提升。

    .. math::
        remainder(input, other) = input - input.div(other, rounding\_mode="floor") * other

    .. note::
        输入不支持复数类型。至少一个输入为tensor，且不能都为布尔型tensor。

    参数：
        - **input** (Union[Tensor, number.Number]) - 除数。
        - **other** (Union[Tensor, number.Number]) - 被除数。

    返回：
        Tensor，经过隐式类型提升和广播。

    异常：
        - **TypeError** - 如果 `input` 和 `other` 不是以下类型之一：(tensor, tensor)，(tensor, number) 或 (number, tensor)。
        - **ValueError** - 如果 `input` 和 `other` 不能被广播。
