mindspore.ops.xdivy
====================

.. py:function:: mindspore.ops.xdivy(x, y)

    `x` 和 `y` 逐元素相除。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    .. note::
        - 支持广播，支持隐式类型转换、类型提升。
        - 当 `x` 和 `y` 的数据类型都为复数时，必须同时为complex64或complex128。
        - `x` 和 `y` 不能同时为bool类型。

    参数：
        - **x** (Union[Tensor, Number, bool]) - 被除数（分子）。
        - **y** (Union[Tensor, Number, bool]) - 除数（分母）。

    返回：
        Tensor
