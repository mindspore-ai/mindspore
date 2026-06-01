mindspore.ops.scatter_nd_div
============================

.. py:function:: mindspore.ops.scatter_nd_div(input_x, indices, updates, use_locking=False)

    `mindspore.ops.scatter_nd_div` 从2.9.0版本开始已被弃用，并将在未来版本中被移除。

    根据指定索引和更新值对 `input_x` 进行稀疏除法更新。

    .. math::
        \text{input_x}[\text{indices}[i, ..., j]] \mathrel{/}= \text{updates}[i, ..., j]

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
