mindspore.ops.assign
=====================

.. py:function:: mindspore.ops.assign(variable, value)

    为网络参数或者tensor赋值。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    支持隐式类型转换、类型提升。

    参数：
        - **variable** (Union[Parameter, Tensor]) - 输入的网络参数或者tensor。
        - **value** (Tensor) - 要分配的值，shape与 `variable` 相同。

    返回：
        Tensor
