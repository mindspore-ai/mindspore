mindspore.ops.add
=================

.. py:function:: mindspore.ops.add(input, other)

    逐元素计算两个输入tensor的和。

    .. math::

        out_{i} = input_{i} + other_{i}

    .. note::
        - 两个输入不能同时为bool类型。[True, Tensor(True), Tensor(np.array([True]))]等都为bool类型。
        - 支持广播，支持隐式类型转换、类型提升。
        - 当输入为tensor时，维度应大于等于1。

    .. warning::
        从2.9.0（不含）之后版本开始，该接口签名将变更为 ``mindspore.ops.add(input, other, *, alpha=1)`` ，
        新增可选关键字参数 `alpha` 。保留现有 `mindspore.ops.add(input, other)` 调用兼容性，
        现有调用保持不变。

    参数：
        - **input** (Union[Tensor, number.Number, bool]) - 第一个输入tensor。
        - **other** (Union[Tensor, number.Number, bool]) - 第二个输入tensor。
    返回：
        Tensor
