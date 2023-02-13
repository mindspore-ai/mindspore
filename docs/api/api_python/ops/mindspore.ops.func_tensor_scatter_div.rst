mindspore.ops.tensor_scatter_div
================================

.. py:function:: mindspore.ops.tensor_scatter_div(input_x, indices, updates)

    根据索引，通过相除运算得到输出Tensor的值。更新后的结果是通过算子output返回，而不是直接原地更新input。

    `indices` 的最后一个轴是每个索引向量的深度。对于每个索引向量， `updates` 中必须有相应的值。 `updates` 的shape应该等于 `input_x[indices]` 的shape。有关更多详细信息，请参见使用用例。

    .. note::
        - 如果 `indices` 的某些值超出范围，则相应的 `updates` 不会更新为 `input_x` ，而不是抛出索引错误。
        - 算子无法处理除0异常，用户需保证 `updates` 中没有0值。

    参数：
        - **input_x** (Tensor) - 输入Tensor。 `input_x` 的维度必须不小于 `indices.shape[-1]` 。
        - **indices** (Tensor) - 输入Tensor的索引，数据类型为int32或int64的。其rank至少为2。
        - **updates** (Tensor) - 指定与 `input_x` 相加操作的Tensor，其数据类型与输入相同。 `updates.shape` 应等于 `indices.shape[:-1] + input_x.shape[indices.shape[-1]:]` 。

    返回：
        Tensor，shape和数据类型与输入 `input_x` 相同。

    异常：
        - **TypeError** - `indices` 的数据类型既不是int32，也不是int64。
        - **ValueError** - `input_x` 的shape长度小于 `indices` 的shape的最后一个维度。
        - **RuntimeError** - `indices` 超出了 `input_x` 的索引范围。
