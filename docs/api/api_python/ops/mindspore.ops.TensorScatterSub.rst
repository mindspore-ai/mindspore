mindspore.ops.TensorScatterSub
===============================

.. py:class:: mindspore.ops.TensorScatterSub

    根据指定的更新值 `input_x` 和输入索引 `indices`，进行减法运算更新输出Tensor的值。当同一索引有不同更新值时，更新的结果将是累积减法的结果。此操作与 :class:`mindspore.ops.ScatterNdSub` 类似，只是更新后的结果是通过算子output返回，而不是直接原地更新input。
    更多参考详见 :func:`mindspore.ops.tensor_scatter_sub`。

    .. math::
        output\left [indices  \right ] = input\_x- update

    输入：
        - **input_x** (Tensor) - 输入Tensor。 `input_x` 的维度必须大于等于indices.shape[-1]。
        - **indices** (Tensor) - 输入Tensor的索引，数据类型为int32或int64，rank必须大于等于2。
        - **updates** (Tensor) - 指定与 `input_x` 相减操作的Tensor，其数据类型与 `input_x` 相同。并且shape应等于 :math:`indices.shape[:-1] + input\_x.shape[indices.shape[-1]:]`。

    输出：
        Tensor，shape和数据类型与输入 `input_x` 相同。
