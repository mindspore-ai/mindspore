mindspore.mint.maximum
=======================

.. py:function:: mindspore.mint.maximum(input, other)

    逐元素计算两个输入tensor中最大值。

    .. math::
        output_i = \max(input_i, other_i)

    .. note::
        - 输入 `input` 和 `other` 遵循隐式类型转换规则，使数据类型保持一致。
        - 当输入是两个tensor时，它们的数据类型不能同时是bool，并保证其shape可以广播。
        - 当输入是一个tensor和一个scalar时，scalar只能是一个常数。
        - 支持广播。
        - 如果一个元素和NaN比较，则返回NaN。

    .. warning::
        如果所有输入都为标量int类型，在Graph模式下，输出为int32类型的Tensor，在PyNative模式下，输出为int64类型的Tensor。

    参数：
        - **input** (Union[Tensor, Number, bool]) - 第一个输入。
        - **other** (Union[Tensor, Number, bool]) - 第二个输入。

    返回：
        Tensor
