mindspore.ops.scatter_div
=========================

.. py:function:: mindspore.ops.scatter_div(input_x, indices, updates)

    根据指定索引和更新值对 `input_x` 进行除法更新。

    .. math::
        \text{input_x}[\text{indices}[i, ..., j], :] \mathrel{/}= \text{updates}[i, ..., j, :]

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    .. note::
        - 支持隐式类型转换、类型提升。
        - 因Parameter对象不支持类型转换，当 `input_x` 为低精度数据类型时，会抛出异常。
        - `updates` 的shape为 `indices.shape + input_x.shape[1:]` 。

    参数：
        - **input_x** (Union[Parameter, Tensor]) - 输入的parameter或tensor。
        - **indices** (Tensor) - 指定索引。
        - **updates** (Tensor) - 更新值。

    返回：
        Tensor
