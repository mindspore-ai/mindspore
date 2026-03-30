mindspore.ops.scatter_update
============================

.. py:function:: mindspore.ops.scatter_update(input_x, indices, updates)

    根据指定输入索引和更新值更新输入tensor的值。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    .. note::
        - 支持隐式类型转换、类型提升。
        - 因Parameter对象不支持类型转换，当 `input_x` 为低精度数据类型时，会抛出异常。
        - 参数 `updates` 的shape为 `indices.shape + input_x.shape[1:]` 。

    若 `indices` 的shape为(i, ..., j)，则

    .. math::
        \text{input_x}[\text{indices}[i, ..., j], :] = \text{updates}[i, ..., j, :]

    参数：
        - **input_x** (Union[Parameter, Tensor]) - 输入的parameter或tensor。
        - **indices** (Tensor) - 更新操作的索引。如果索引中存在重复项，则更新的顺序无法得知。
        - **updates** (Tensor) - 更新值。

    返回：
        Tensor
