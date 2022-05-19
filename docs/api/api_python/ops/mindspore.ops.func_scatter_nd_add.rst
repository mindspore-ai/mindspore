mindspore.ops.scatter_nd_add
============================

.. py:function:: mindspore.ops.scatter_nd_add(input_x, indices, updates, use_locking=False)

    对Tensor中的单个值或切片应用加法操作。

    通过加法操作以及输入索引值，使用给定值来更新Tensor。更新完成后，此操作会输出 `input_x`，以便于使用更新后的值。

    `input_x` 的秩为P，而 `indices` 的秩为Q， `Q >= 2` 。

    `indices` 的shape为 :math:`(i_0, i_1, ..., i_{Q-2}, N)` ， `N <= P` 。`indices` 的最后一个维度（长度为 `N` ）表示沿着 `input_x` 的 `N` 个维度进行切片。

    `updates` 表示秩为 `Q-1+P-N` 的Tensor，其shape为 :math:`(i_0, i_1, ..., i_{Q-2}, shape_N, ..., shape_{P-1})` 。

    `input_x` 和 `updates` 的输入符合隐式类型转换规则，使数据类型一致。如果它们具有不同的数据类型，则较低优先级的数据类型将转换为相对优先级最高的数据类型。

    **参数：**

    - **input_x** (Parameter) - 目标Tensor，数据类型为Parameter。其维度为 :math:`(N,*)` ，其中 :math:`*` 表示任意数量的附加维度。
    - **indices** (Tensor) - 指定新Tensor中散布的索引，数据类型为int32。索引的秩须至少为2，并且 `indices_shape[-1] <= len(shape)` 。
    - **updates** (Tensor) - 用来与 `input_x` 进行最小运算的Tensor，数据类型与 `input_x` 相同，其维度为 `indices.shape[:-1] + x.shape[indices.shape[-1]:]` 。
    - **use_locking** (bool) - 是否通过锁保护分配，默认值：False。

    **返回：**

    Tensor，更新后的Tensor，数据类型与输入 `input_x` 相同，维度与输入 `input_x` 相同。

    **异常：**

    - **TypeError** - 如果 'use_locking` 数据类型不是bool。
    - **TypeError** - 如果 `indices` 数据类型不是int32。
    - **ValueError** - 如果 `updates` 的维度不等于 `indices.shape[:-1] + x.shape[indices.shape[-1]:]`。
    - **RuntimeError** - 如果 `input_x` 和 `updates` 需要数据类型转换且数据类型依然无法完成隐式数据类型转换。
