mindspore.ops.assign_add
=========================

.. py:function:: mindspore.ops.assign_add(variable, value)

    通过加法运算更新网络参数。

    `variable` 和 `value` 遵循隐式类型转换规则，使数据类型一致。如果它们具有不同的数据类型，则低精度数据类型将转换为相对最高精度的数据类型。如果 `value` 是数值型，则会自动转换为Tensor，数据类型与计算中涉及的Tensor的数据类型保持一致。

    .. note::
        由于 `variable` 是数据类型为Parameter，数据类型不能更改，因此只允许 `value` 的数据类型转换为 `variable` 的数据类型。而且不同设备支持的转换类型会有所不同，建议在使用此函数时使用相同的数据类型。

    参数：
        - **variable** (Parameter) - 网络参数。shape为 :math:`(N,*)` ，其中 :math:`*` 表示任意数量的附加维度，其秩应小于8。
        - **value** (Tensor) - 要和 `variable` 相加的值，shape必须与 `variable` 相同。建议在使用此函数时使用相同的数据类型。
        
    返回：
        Tensor，数据类型和shape与输入 `variable` 相同。
        
    异常：
        - **TypeError** - 如果 `value` 既不是数值型也不是Tensor。
        - **RuntimeError** - 如果 `variable` 和 `value` 不支持参数的数据类型转换。
