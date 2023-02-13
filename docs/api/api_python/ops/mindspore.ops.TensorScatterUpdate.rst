mindspore.ops.TensorScatterUpdate
==================================

.. py:class:: mindspore.ops.TensorScatterUpdate

    根据指定的更新值和输入索引，通过更新操作更新输入Tensor的值。此操作几乎等同于使用 :class:`mindspore.ops.ScatterNdUpdate` ，只是更新操作应用到 `input_x` Tensor而不是0。

    `indices` 的rank至少要为2，最后一个轴表示每个索引向量的深度。对于每个索引向量， `update` 中必须有相应的值。如果每个索引Tensor的深度与 `input_x` 的rank匹配，则每个索引向量对应于 `input_x` 中的Scalar，并且每次更新都会更新一个Scalar。如果每个索引Tensor的深度小于 `input_x` 的rank，则每个索引向量对应于 `input_x` 中的切片，并且每次更新都会更新一个切片。

    更新的顺序是不确定的，这意味着如果 `indices` 中有多个索引向量对应于同一位置，则输出中该位置值是不确定的。

    输入：
        - **input_x** (Tensor) - TensorScatterUpdate的输入，任意维度的Tensor。其数据类型为数值型。 `input_x` 的维度必须不小于indices.shape[-1]。
        - **indices** (Tensor) - 输入Tensor的索引，数据类型为int32或int64。其rank必须至少为2。
        - **update** (Tensor) - 指定与 `input_x` 做更新操作的Tensor，其数据类型与输入相同。:math:`update.shape = indices.shape[:-1]+input_x.shape[indices.shape[-1]:]`。

    输出：
        Tensor，shape和数据类型与输入 `input_x` 相同。

    异常：
        - **TypeError** - `indices` 的数据类型既不是int32，也不是int64。
        - **ValueError** - `input_x` 的shape长度小于 `indices` 的shape的最后一个维度。
        - **ValueError** - `input_x` 的值与输入 `indices` 不匹配。
        - **RuntimeError** - `indices` 超出了 `input_x` 的索引范围。