mindspore.ops.fmod
===================

.. py:function:: mindspore.ops.fmod(x, other)

    计算除法运算 x/other 的浮点余数。

    .. math::
        out = x - n * other

    其中 :math:`n` 是 :math:`x/other` 结果中的整数部分。
    返回值的符号和 `x` 相同，在数值上小于 `other` 。

    参数：
        - **x** (Union[Tensor, Number]) - 被除数。
        - **other** (Union[Tensor, Number]) - 除数。

    返回：
        Tensor，输出的shape与广播后的shape相同，数据类型取两个输入中精度较高或数字较高的。

    异常：
        - **TypeError** - `x` 和 `other` 都不是Tensor。
