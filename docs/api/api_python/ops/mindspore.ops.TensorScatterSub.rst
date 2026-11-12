mindspore.ops.TensorScatterSub
===============================

.. py:class:: mindspore.ops.TensorScatterSub

    `mindspore.ops.TensorScatterSub` 从2.10.0版本开始已被弃用，并将在未来版本中被移除。

    根据指定的更新值 `input_x` 和输入索引 `indices`，进行减法运算更新输入Tensor的值。当同一索引有不同更新值时，更新的结果将是累积减法的结果。更新结果写入新的输出 Tensor，输入保持不变。

    .. code-block:: python

        # 遍历所有索引
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                ...
                for k in range(indices.shape[-2]):  # 最后一维是坐标维度
                    # 获取当前索引组合
                    index_tuple = (i, j, ..., k)
                    # 获取目标位置
                    target_index = indices[index_tuple]
                    # 获取对应更新值
                    update_value = updates[index_tuple]
                    # 执行减法操作
                    output[target_index] -= update_value

    更多参考详见 :func:`mindspore.ops.tensor_scatter_sub`。

    输入：
        - **input_x** (Tensor) - 输入Tensor。 `input_x` 的维度必须大于等于indices.shape[-1]。
        - **indices** (Tensor) - 输入Tensor的索引，数据类型为int32或int64，rank必须大于等于2。
        - **updates** (Tensor) - 指定与 `input_x` 相减操作的Tensor，其数据类型与 `input_x` 相同。并且shape应等于 :math:`indices.shape[:-1] + input\_x.shape[indices.shape[-1]:]`。

    输出：
        Tensor，shape和数据类型与输入 `input_x` 相同。
