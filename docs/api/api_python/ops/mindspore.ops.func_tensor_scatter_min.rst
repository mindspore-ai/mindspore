mindspore.ops.tensor_scatter_min
==================================

.. py:function:: mindspore.ops.tensor_scatter_min(input_x, indices, updates)

    根据指定的更新值 `updates` 和输入索引 `indices` ，使用最小值运算更新 `input_x`，输出结果通过新的输出Tensor返回。

    索引 `indices` 的最后一个轴是每个索引向量的深度。对于每个索引向量， `updates` 中必须有相应的值。 并且其shape应该等于 `input_x[indices]` 的shape。更多详细信息，请参见下方样例。

    .. math::
        output\left [indices  \right ] = \min(input\_x, update)

    .. note::
        如果 `indices` 中的值超出输入 `input_x` 索引范围：

        - GPU平台上相应的 `updates` 不会更新到 `input_x` 且不会抛出索引错误。
        - CPU平台上直接抛出索引错误。
        - Ascend平台不支持越界检查，若越界可能会造成未知错误。
        
    参数：
        - **input_x** (Tensor) - 输入Tensor。 `input_x` 的维度必须不小于 `indices.shape[-1]` 。
        - **indices** (Tensor) - 输入Tensor的索引，数据类型为int32或int64。其rank至少为2。
        - **updates** (Tensor) - 指定与 `input_x` 取最小值操作的Tensor，其数据类型与 `input_x` 相同。并且其shape应等于 :math:`indices.shape[:-1] + input\_x.shape[indices.shape[-1]:]` 。

    返回：
        Tensor，shape和数据类型与输入 `input_x` 相同。

    异常：
        - **TypeError** - `indices` 的数据类型不为int32或int64。
        - **ValueError** - `input_x` 的rank小于 `indices.shape` 的最后一维。
        - **RuntimeError** - 在CPU平台中，`indices` 中的值超出了 `input_x` 的索引范围。
