mindspore.ops.tensor_scatter_div
================================

.. py:function:: mindspore.ops.tensor_scatter_div(input_x, indices, updates)

    返回一个新tensor，根据指定索引和更新值对 `input_x` 进行除法更新。

    .. math::
        output\left [indices  \right ] = input\_x \div updates

    .. note::
        - 如果 `indices` 中的值超出输入 `input_x` 索引范围：

          - GPU平台上相应的 `updates` 不会更新到 `input_x` 且不会抛出索引错误。
          - CPU平台上直接抛出索引错误。
          - Ascend平台不支持越界检查，若越界可能会造成未知错误。
        - 算子无法处理除0异常，用户需保证 `updates` 中没有0值。

    参数：
        - **input_x** (Tensor) - 输入tensor。
        - **indices** (Tensor) - 指定索引。
        - **updates** (Tensor) - 更新值。

    返回：
        Tensor
