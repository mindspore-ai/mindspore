mindspore.ops.tensor_scatter_add
==================================

.. py:function:: mindspore.ops.tensor_scatter_add(input_x, indices, updates)

    返回一个新tensor，根据指定索引和更新值对 `input_x` 进行加法更新。

    .. math::
        output\left [indices  \right ] = input\_x + updates

    下图是tensor_scatter_add计算过程的例子：

    .. image:: ../images/TensorScatterAdd.png
        :align: center

    .. note::
        如果 `indices` 中的值超出输入 `input_x` 索引范围：

        - GPU平台上相应的 `updates` 不会更新到 `input_x` 且不会抛出索引错误。
        - CPU平台上直接抛出索引错误。
        - Ascend平台不支持越界检查，若越界可能会造成未知错误。

    参数：
        - **input_x** (Tensor) - 输入tensor。 `input_x` 的维度必须不小于indices.shape[-1]。
        - **indices** (Tensor) - 输入tensor的索引，数据类型为mindspore.int32或mindspore.int64。其rank必须至少为2。
        - **updates** (Tensor) - 指定与 `input_x` 相加的Tensor，其数据类型与 `input_x` 相同，并且shape应等于 :math:`indices.shape[:-1] + input\_x.shape[indices.shape[-1]:]` 。

    返回：
        Tensor
