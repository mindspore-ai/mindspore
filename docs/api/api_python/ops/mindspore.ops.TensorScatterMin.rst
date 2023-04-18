mindspore.ops.TensorScatterMin
===============================

.. py:class:: mindspore.ops.TensorScatterMin

    根据指定的更新值和输入索引，计算原值与更新值的较小值并更新原值，返回更新后的Tensor。

    更多参考详见 :func:`mindspore.ops.tensor_scatter_min`。

    输入：
        - **input_x** (Tensor) - 输入Tensor。 `input_x` 的维度必须不小于indices.shape[-1]。
        - **indices** (Tensor) - 输入Tensor的索引，数据类型为int32或int64。其rank必须至少为2。
        - **updates** (Tensor) - 指定与 `input_x` 取最小值操作的Tensor，其数据类型与输入相同。updates.shape应该等于indices.shape[:-1] + input_x.shape[indices.shape[-1]:]。

    输出：
        Tensor，shape和数据类型与输入 `input_x` 相同。
