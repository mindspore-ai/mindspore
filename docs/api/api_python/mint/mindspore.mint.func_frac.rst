mindspore.mint.frac
=====================

.. py:function:: mindspore.mint.frac(input)

    计算 `input` 中每个元素的小数部分。

    .. math::
        out_i = input_i - \lfloor |input_i| \rfloor * sgn(input_i)

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入tensor。

    返回：
        Tensor，其类型和shape与 `input` 相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
