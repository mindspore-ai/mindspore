mindspore.ops.Assign
====================

.. py:class:: mindspore.ops.Assign

    为网络参数赋值。

    输入 `variable` 和 `value` 会通过隐式数据类型转换使数据类型保持一致。如果数据类型不同，低精度的数据类型会被转换到高精度的数据类型。

    **输入：**

    - **variable** (Parameter) - 待赋值的网络参数，shape: math:`(N,*)` ，其中 :math:`*` 表示任何数量的附加维度。其轶应小于8。
    - **value** (Tensor) - 被赋给网络参数的值，和 `variable` 有相同的shape。

    **输出：**

    Tensor，shape和dtype与 `variable` 相同。

    **异常：**

    - **TypeError** - `variable` 不是Parameter类型。
    - **TypeError** - 'value' 不是Tensor。
    - **RuntimeError** - `variable` 与 `value` 之间的类型转换不被支持。
