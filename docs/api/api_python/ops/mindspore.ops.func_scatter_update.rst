mindspore.ops.scatter_update
============================

.. py:function:: mindspore.ops.scatter_update(input_x, indices, updates)

    使用给定的更新值和输入索引更新输入Tensor的值。

    若 `indices` 的shape为(i, ..., j)，则

    .. math::
        \text{input_x}[\text{indices}[i, ..., j], :]= \text{updates}[i, ..., j, :]

    输入的 `input_x` 和 `updates` 遵循隐式类型转换规则，以确保数据类型一致。如果它们具有不同的数据类型，则低精度数据类型将转换为高精度数据类型。因Parameter对象不支持类型转换，当 `input_x` 为低精度数据类型时，会抛出异常。

    参数：
        - **input_x** (Parameter) - scatter_update的输入，任意维度的Parameter。
        - **indices** (Tensor) - 指定更新操作的索引。数据类型为int32或者int64。如果索引中存在重复项，则更新的顺序无法得知。
        - **updates** (Tensor) - 指定与 `input_x` 更新操作的Tensor，其数据类型与 `input_x` 相同，shape为 `indices.shape + input_x.shape[1:]` 。

    返回：
        Tensor，shape和数据类型与输入 `input_x` 相同。

    异常：
        - **TypeError** - `indices` 不是int32或者int64。
        - **ValueError** - `updates` 的shape不等于 `indices.shape + input_x.shape[1:]` 。
        - **RuntimeError** - 当 `input_x` 和 `updates` 类型不一致，需要进行类型转换时，如果 `updates` 不支持转成参数 `input_x` 需要的数据类型，就会报错。
