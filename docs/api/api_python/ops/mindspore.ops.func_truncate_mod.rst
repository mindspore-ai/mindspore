mindspore.ops.truncate_mod
==========================

.. py:function:: mindspore.ops.truncate_mod(x, y)

    逐元素取模。

    输入 `x` 和 `y` 应能遵循隐式类型转换规则使数据类型一致。
    输入必须为两个Tensor或一个Tensor和一个标量。
    当输入为两个Tensor时，数据类型不能同时为bool类型，支持shape广播。
    当输入是一个Tensor和一个标量时，标量只能是一个常数。

    .. warning::
        - 输入数值不能为0。
        - 当输入含有超过2048个元素，该操作不能保证千分之二的精度要求。
        - 由于架构不同，该算子在NPU和CPU上的计算结果可能不一致。
        - 若shape为（D1、D2...、Dn），则D1*D2...*DN<=1000000，n<=8。

    参数：
        - **x** (Union[Tensor, numbers.Number, bool]) - Number或bool类型的Tensor。
        - **y** (Union[Tensor, numbers.Number, bool]) - Number或bool类型的Tensor。

    返回：
        Tensor，shape为输入进行广播后的shape，数据类型为两个输入中精度较高的输入的类型。

    异常：
        - **TypeError** - `x` 和 `y` 数据类型不是以下之一：Tensor、Number、bool。
        - **TypeError** - `x` 和 `y` 均不是Tensor。
        - **ValueError** - `x` 和 `y` 的shape无法进行广播转换。
