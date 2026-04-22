mindspore.ops.scatter_nd_min
============================

.. py:function:: mindspore.ops.scatter_nd_min(input_x, indices, updates, use_locking=False)

    根据指定索引和更新值对 `input_x` 进行稀疏最小值更新。

    .. math::
        \text{input_x}[\text{indices}[i, ..., j]]
        = \min(\text{input_x}[\text{indices}[i, ..., j]], \text{updates}[i, ..., j])

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    .. note::
        - 支持隐式类型转换、类型提升。
        - `indices` 的维度至少为2，并且 `indices.shape[-1] <= len(indices.shape)` 。
        - `updates` 的shape为 `indices.shape[:-1] + input_x.shape[indices.shape[-1]:]` 。

    参数：
        - **input_x** (Union[Parameter, Tensor]) - 输入的parameter或tensor。
        - **indices** (Tensor) - 指定索引。
        - **updates** (Tensor) - 更新值。
        - **use_locking** (bool) - 是否启用锁保护。默认 ``False`` 。

    返回：
        Tensor
