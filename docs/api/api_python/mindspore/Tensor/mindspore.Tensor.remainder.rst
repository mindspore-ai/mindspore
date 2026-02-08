mindspore.Tensor.remainder
===========================

.. py:method:: mindspore.Tensor.remainder(other) -> Tensor

    逐元素计算 `self` 除以 `other` 后的余数。结果与除数同号且绝对值小于除数的绝对值。

    支持广播和隐式数据类型提升。

    .. code:: python

        remainder(input, other) == input - input.div(other, rounding_mode="floor") * other

    .. note::
        输入不支持复数类型。至少一个输入为Tensor，且不能都为bool型Tensor。
        被除数 `self` 是数据类型为 `number <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 或 `bool <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 的Tensor。

    参数：
        - **other** (Union[Tensor, numbers.Number, bool]) - 除数为数值型、bool或数据类型为数值型或bool的Tensor。当除数是Tensor时，则被除数是数值型、bool或数据类型为数值型或bool的Tensor。

    返回：
        Tensor，经过隐式类型提升和广播。

    异常：
        - **TypeError** - 如果 `self` 和 `other` 不是以下类型之一：(Tensor, Tensor)、(Tensor, Number)、(Tensor, bool)、(Number, Tensor)或(bool, Tensor)。
        - **ValueError** - 如果 `self` 和 `other` 不能被广播。

    .. py:method:: mindspore.Tensor.remainder(divisor) -> Tensor
        :noindex:

    逐元素计算 `self` 除以 `divisor` 的余数。

    `self` 和 `divisor` 的输入遵守隐式类型转换规则，以使数据类型一致。输入必须是两个Tensor或者一个Tensor和一个Scalar。当输入是两个Tensor时，两个dtype都不能是bool类型，shape可以广播。当输入是Tensor和Scalar时，这个Scalar只能是常数。

    .. code:: python

        remainder(input, other) == input - input.div(other, rounding_mode="floor") * other

    .. warning::
        - 当输入元素超过2048时，可能会有精度问题。
        - 在Ascend和CPU上的计算结果可能不一致。
        - 如果shape表示为(D1,D2…Dn)，那么D1 \ * D2……\ * DN <= 1000000，n <= 8。

    .. note::
        第一个输入 `self` 为dtype是Number的Tensor。

    参数：
        - **divisor** (Union[Tensor, numbers.Number, bool]) - 当第一个输入是一个Tensor时，第二个输入可以是Number、bool或者dtype是Number的Tensor。

    返回：
        Tensor，具有和其中一个输入广播后相同的shape，数据类型是两个输入中精度较高的数据类型。

    异常：
        - **TypeError** - `self` 和 `divisor` 的类型不是Tensor、Number或bool。
        - **ValueError** - `self` 和 `divisor` 的shape不能广播为对方的shape。
