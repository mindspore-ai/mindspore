mindspore.ops.TruncateDiv
=========================

.. py:class:: mindspore.ops.TruncateDiv

    对于整数类型，将第一个输入Tensor与第二个输入Tensor逐元素相除，结果将向0取整。

    输入 `x` 和 `y` 应能遵循隐式类型转换规则使数据类型一致。
    输入必须为两个Tensor或一个Tensor和一个标量。
    当输入为两个Tensor时，数据类型不能同时为bool类型。
    当输入是一个Tensor和一个标量时，标量只能是一个常数。

    .. note::
        支持shape广播。

    输入：
        - **x** (Union[Tensor, Number, bool]) - Number或bool类型的Tensor。
        - **y** (Union[Tensor, Number, bool]) - Number或bool类型的Tensor。`x` 和 `y` 不能同时为bool类型。

    输出：
        Tensor，shape为输入广播后的shape，数据类型为两个输入中精度较高的输入的类型。

    异常：
        - **TypeError** - `x` 和 `y` 数据类型不是以下之一：Tensor、Number或bool。
