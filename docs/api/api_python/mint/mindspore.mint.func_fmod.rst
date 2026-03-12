mindspore.mint.fmod
=====================

.. py:function:: mindspore.mint.fmod(input, other) -> Tensor

    计算除法运算 input/other 的浮点余数。

    .. math::
        out = input - n * other

    其中 :math:`n` 是 :math:`input/other` 结果中的整数部分。
    返回值的符号和 `input` 相同，在数值上小于 `other` 。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 被除数。
        - **other** (Union[Tensor, Number]) - 除数。

    返回：
        Tensor，输出的shape与广播后的shape相同，数据类型取两个输入中精度较高或位数较高的。

    异常：
        - **TypeError** - `input` 不是Tensor。
