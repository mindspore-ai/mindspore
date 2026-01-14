mindspore.ops.tensor_scatter_max
===================================

.. py:function:: mindspore.ops.tensor_scatter_max(input_x, indices, updates)

    返回一个新tensor，根据指定索引和更新值对 `input_x` 进行最大值更新。

    .. math::
        output\left [indices  \right ] = \max(input\_x, updates)

    .. note::
        如果 `indices` 中的值超出输入 `input_x` 索引范围：

        - GPU平台上相应的 `updates` 不会更新到 `input_x` 且不会抛出索引错误。
        - CPU平台上直接抛出索引错误。
        - Ascend平台不支持越界检查，若越界可能会造成未知错误。

    参数：
        - **input_x** (Tensor) - 输入tensor。
        - **indices** (Tensor) - 指定索引，其秩至少为2。
        - **updates** (Tensor) - 更新值。

    返回：
        Tensor

