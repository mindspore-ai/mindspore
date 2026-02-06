mindspore.ops.tensor_scatter_mul
================================

.. py:function:: mindspore.ops.tensor_scatter_mul(input_x, indices, updates)

    根据指定索引和更新值对 `input_x` 进行乘法更新。

    .. math::
        output\left [indices  \right ] = input\_x\times  updates

    .. note::
        - 如果 `indices` 的某些值超出 `input_x` 的维度范围，则相应的 `updates` 不会更新到 `input_x` ，而不是抛出索引错误。

    参数：
        - **input_x** (Tensor) - 输入tensor。
        - **indices** (Tensor) - 指定索引，其秩至少为2。
        - **updates** (Tensor) - 更新值。

    返回：
        Tensor
