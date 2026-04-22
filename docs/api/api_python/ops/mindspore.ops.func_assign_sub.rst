mindspore.ops.assign_sub
========================

.. py:function:: mindspore.ops.assign_sub(variable, value)

    通过减法运算更新网络参数或者tensor。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    支持隐式类型转换、类型提升。

    参数：
        - **variable** (Union[Parameter, Tensor]) - 输入的网络参数或者tensor。
        - **value** (Tensor) - 从 `variable` 减去的值。

    返回：
        Tensor
