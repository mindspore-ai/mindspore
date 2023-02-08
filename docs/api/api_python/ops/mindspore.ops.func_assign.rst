mindspore.ops.assign
=====================

.. py:function:: mindspore.ops.assign(variable, value)

    为网络参数赋值。

    `variable` 和 `value` 遵循隐式类型转换规则，使数据类型一致。如果它们具有不同的数据类型，则低精度数据类型将转换为相对最高精度的数据类型。

    参数：
        - **variable** (Parameter) - 网络参数。 :math:`(N,*)` ，其中 :math:`*` 表示任意数量的附加维度，其秩应小于8。
        - **value** (Tensor) - 要分配的值，shape与 `variable` 相同。

    返回：
        Tensor，数据类型和shape与 `variable` 相同。

    异常：
        - **TypeError** - 如果 `variable` 不是Parameter。
        - **TypeError** - 如果 `value` 不是Tensor。
        - **RuntimeError** - 如果 `variable` 和 `value` 不支持参数的数据类型转换。
