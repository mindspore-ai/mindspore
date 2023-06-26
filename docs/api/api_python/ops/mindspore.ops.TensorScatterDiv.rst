mindspore.ops.TensorScatterDiv
==============================

.. py:class:: mindspore.ops.TensorScatterDiv

    根据指定的更新值 `updates` 和输入索引 `indices` ，进行除法运算更新输入Tensor的值。当同一索引有不同更新值时，更新的结果将是累积除法的结果。此操作但更新后的结果是返回一个新的输出Tensor，而不是直接原地更新 `input_x` 。

    更多参考相见 :func:`mindspore.ops.tensor_scatter_div`。

    .. math::
        output\left [indices  \right ] = input\_x \div update

    输入：
        - **input_x** (Tensor) - 输入Tensor。 `input_x` 的维度必须不小于indices.shape[-1]。
        - **indices** (Tensor) - 输入Tensor的索引，数据类型为int32或int64。其rank必须至少为2。
        - **updates** (Tensor) - 指定与 `input_x` 相除的Tensor，其数据类型与 `input_x` 相同，并且shape应等于 :math:`indices.shape[:-1] + input\_x.shape[indices.shape[-1]:]` 。

    输出：
        Tensor，shape和数据类型与输入 `input_x` 相同。
