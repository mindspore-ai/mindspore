mindspore.mint.bitwise_xor
=============================

.. py:function:: mindspore.mint.bitwise_xor(input, other)

    逐元素执行两个Tensor的异或运算。

    .. note::
        参数 `input` 和 `other` 遵循隐式类型转换规则，使数据类型保持一致。如果两参数数据类型不一致，则低精度类型会被转换成较高精度类型。

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **other** (Tensor, Number.number) - 输入Tensor或常数，shape与 `input` 相同，或能与 `input` 的shape广播。

    返回：
        Tensor，与广播后的输入shape相同，和 `input` 数据类型相同。
