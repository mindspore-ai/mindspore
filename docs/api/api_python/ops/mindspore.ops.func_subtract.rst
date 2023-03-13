mindspore.ops.subtract
=======================

.. py:function:: mindspore.ops.subtract(input, other, *, alpha=1)

    对Tensor进行逐元素的减法。

    .. math::
        output[i] = input[i] - alpha * y[i]

    参数：
        - **input** (Union[Tensor, number.Number]) - 参与减法的Tensor或者Number。
        - **other** (Union[Tensor, number.Number]) - 参与减法的Tensor或者Number。

    关键字参数：
        - **alpha** (Number) - :math:`other` 的乘数。默认值：1。

    返回：
        Tensor，shape与广播后的shape相同，数据类型为输入中精度较高的类型。

    异常：
        - **TypeError** - `input` 或 `other` 不是Tensor、number.Number。
