mindspore.ops.expand_dims
=========================

.. py:function:: mindspore.ops.expand_dims(input_x, axis)

    为输入tensor新增额外的轴。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    .. note::
        - `input_x` 的维度应该大于等于1。
        - 如果指定的 `axis` 是负数，那么它会从后往前，从1开始计算index。

    参数：
        - **input_x** (Tensor) - 输入tensor。
        - **axis** (int) - 新增的轴。仅接受常量。

    返回：
        Tensor
