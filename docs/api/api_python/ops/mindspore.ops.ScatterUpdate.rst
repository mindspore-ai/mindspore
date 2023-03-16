mindspore.ops.ScatterUpdate
============================

.. py:class:: mindspore.ops.ScatterUpdate(use_locking=True)

    使用给定的更新值和输入索引更新输入Tensor的值。

    使用给定的值来更新张量值，以及输入指数。

    若 `indices` 的shape为(i, ..., j)，则

    .. math::
        \text{input_x}[\text{indices}[i, ..., j], :] = \text{updates}[i, ..., j, :]

    输入的 `input_x` 和 `updates` 遵循隐式类型转换规则，以确保数据类型一致。如果它们具有不同的数据类型，则低精度数据类型将转换为高精度数据类型。当需要转换Parameter的数据类型时，会抛出RuntimeError异常。

    参数：
        - **use_locking** (bool) - 表示是否使用锁来保护。默认值：True。

    输入：
        - **input_x** (Parameter) - ScatterUpdate的输入，shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。
        - **indices** (Tensor) - 指定更新操作的索引。数据类型为int32。如果索引中存在重复项，则更新的顺序无法得知。
        - **updates** (Tensor) - 指定与 `input_x` 更新操作的Tensor，其数据类型与 `input_x` 相同，shape为 `indices.shape + input_x.shape[1:]` 。

    输出：
        Tensor，shape和数据类型与输入 `input_x` 相同。

    异常：
        - **TypeError** - `use_locking` 不是bool。
        - **TypeError** - `indices` 不是int32。
        - **ValueError** - `updates` 的shape不等于 `indices.shape + input_x.shape[1:]` 。
        - **RuntimeError** - 当 `input_x` 和 `updates` 类型不一致，需要进行类型转换时，如果 `updates` 不支持转成参数 `input_x` 需要的数据类型，就会报错。