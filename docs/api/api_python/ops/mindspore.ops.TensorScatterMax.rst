mindspore.ops.TensorScatterMax
===============================

.. py:class:: mindspore.ops.TensorScatterMax

    根据指定的更新值 `updates` 和输入索引 `indices` ，通过最大值运算将结果赋值到输出Tensor中。

    更多参考详见 :func:`mindspore.ops.tensor_scatter_max`。

    输入：
        - **input_x** (Tensor) - 输入Tensor。 `input_x` 的维度必须不小于indices.shape[-1]。
        - **indices** (Tensor) - 输入Tensor的索引，数据类型为int32或int64。其rank必须至少为2。
        - **updates** (Tensor) - 指定与 `input_x` 取最大值的Tensor，其数据类型与 `input_x` 相同，并且shape应等于 :math:`indices.shape[:-1] + input\_x.shape[indices.shape[-1]:]` 。

    输出：
        Tensor，shape和数据类型与输入 `input_x` 相同。
