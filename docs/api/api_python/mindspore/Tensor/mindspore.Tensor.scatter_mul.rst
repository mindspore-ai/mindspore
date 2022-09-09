mindspore.Tensor.scatter_mul
============================

.. py:method:: mindspore.Tensor.scatter_mul(indices, updates)

    根据指定的索引，通过乘法进行计算，将结果赋值到输出Tensor中。更新后的结果是通过算子output返回，而不是直接原地更新当前Tensor。

    `indices` 的最后一个轴是每个索引向量的深度。对于每个索引向量， `updates` 中必须有相应的值。 `updates` 的shape应该等于 `input_x[indices]` 的shape。其中 `input_x` 指当前Tensor。有关更多详细信息，请参见使用用例。

    .. note::
        - 如果 `indices` 的某些值超出范围，则相应的 `updates` 不会更新为当前Tensor，而不是抛出索引错误。

    参数：
        - **indices** (Tensor) - 该Tensor的索引，数据类型为int32或int64的。其rank必须至少为2。
        - **updates** (Tensor) - 指定与当前Tensor相加操作的Tensor，其数据类型与输入相同。updates.shape应等于 `indices.shape[:-1] + input_x.shape[indices.shape[-1]:]`，其中 `input_x` 代指当前Tensor本身。

    返回：
        Tensor，shape和数据类型与该Tensor相同。

    异常：
        - **TypeError** - `indices` 的数据类型不是int32，也不是int64。
        - **ValueError** - Tensor的shape长度小于 `indices` 的shape的最后一个维度。