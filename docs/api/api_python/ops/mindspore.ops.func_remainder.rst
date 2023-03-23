mindspore.ops.remainder
=======================

.. py:function:: mindspore.ops.remainder(input, other)

    逐元素计算第一个元素除第二个元素的余数。

    `input` 和 `other` 的输入遵守隐式类型转换规则，以使数据类型一致。输入必须是两个Tensor或者一个Tensor和一个Scalar。当输入是两个Tensor时，两个dtype都不能是bool类型，shape可以广播。当输入是Tensor和Scalar时，这个Scalar只能是常数。

    .. math::
        remainder(input, other) == input - input.div(other, rounding_mode="floor") * other

    .. warning::
        - 当输入元素超过2048时，可能会有精度问题。
        - 在Ascend和CPU上的计算结果可能不一致。
        - 如果shape表示为(D1,D2…Dn)，那么D1 \ * D2……\ * DN <= 1000000，n <= 8。

    参数：
        - **input** (Union[Tensor, numbers.Number, bool]) - 第一个输入可以是Number，bool或者dtype是Number的Tensor。
        - **other** (Union[Tensor, numbers.Number, bool]) - 当第一个输入是一个Tensor时，第二个输入可以是Number、bool或者dtype是Number的Tensor。

    返回：
        Tensor，具有和其中一个输入广播后相同的shape，数据类型是两个输入中精度较高的数据类型。

    异常：
        - **TypeError** - `input` 和 `other` 的类型不是Tensor，number或bool。
        - **ValueError** - `input` 和 `other` 的shape不能广播成对方的shape。
