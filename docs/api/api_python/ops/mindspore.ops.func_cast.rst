mindspore.ops.cast
===================

.. py:function:: mindspore.ops.cast(input, dtype)

    转换输入tensor的数据类型。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    .. note::
        将复数转换为bool类型的时候，不考虑复数的虚部，只要实部不为零，返回 ``True`` ，否则返回 ``False`` 。

    参数：
        - **input** (Union[Tensor, Number]) - 输入tensor或者数值型数据。
        - **dtype** (dtype.Number) - 转换后的数据类型。仅支持常量值。

    返回：
        Tensor
