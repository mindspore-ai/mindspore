mindspore.ops.tensor_scatter_max
===================================

.. py:function:: mindspore.ops.tensor_scatter_max(input_x, indices, updates)

    根据指定的更新值和输入索引，通过最大值运算，输出结果以Tensor形式返回。

    索引的最后一个轴是每个索引向量的深度。对于每个索引向量， `updates` 中必须有相应的值。 `updates` 的shape应该等于 `input_x[indices]` 的shape。

    .. note::
        如果 `indices` 的某些值超出范围，则 `input_x` 不会更新相应的 `updates`，同时也不会抛出索引错误。

    参数：
        - **input_x** (Tensor) - 输入Tensor。 `input_x` 的维度必须不小于indices.shape[-1]。
        - **indices** (Tensor) - 输入Tensor的索引，数据类型为int32或int64。其rank必须至少为2。
        - **updates** (Tensor) - 指定与 `input_x` 取最小值操作的Tensor，其数据类型与输入相同。updates.shape应该等于indices.shape[:-1] + input_x.shape[indices.shape[-1]:]。

    返回：
        Tensor，shape和数据类型与输入 `input_x` 相同。

    异常：
        - **TypeError** - `indices` 的数据类型既不是int32，也不是int64。
        - **ValueError** - `input_x` 的shape长度小于 `indices` 的shape的最后一个维度。
        - **RuntimeError** - `indices` 超出了 `input_x` 的索引范围。
