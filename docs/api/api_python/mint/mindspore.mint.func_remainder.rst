mindspore.mint.remainder
========================

.. py:function:: mindspore.mint.remainder(input, other) -> Tensor

    逐元素计算 `input` 除以 `other` 后的余数。结果与除数同号且绝对值小于除数的绝对值。

    支持广播和隐式数据类型提升。

    .. code:: python

        remainder(input, other) == input - input.div(other, rounding_mode="floor") * other

    .. note::
        输入不支持复数类型。至少一个输入必须为tensor，且不能都为布尔型tensor。

    参数：
        - **input** (Union[Tensor, numbers.Number, bool]) - 被除数为数值型、bool或数据类型为 `number <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 或 `bool <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 的Tensor。
        - **other** (Union[Tensor, numbers.Number, bool]) - 除数为数值型、bool或数据类型为数值型或bool的Tensor。当除数是Tensor时，则被除数是数值型、bool或数据类型为数值型或bool的Tensor。当除数是Scalar时，则被除数必须是数据类型为数值型或bool的Tensor。

    返回：
        Tensor，其数据类型经过隐式类型提升和广播。

    异常：
        - **TypeError** - 如果 `input` 和 `other` 不是以下类型之一：(tensor, tensor)，(tensor, number)，(tensor, bool)，(number, tensor) 或 (bool, tensor)。
        - **ValueError** - 如果 `input` 和 `other` 不能被广播。
