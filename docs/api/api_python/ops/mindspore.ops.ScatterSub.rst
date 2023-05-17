mindspore.ops.ScatterSub
=========================

.. py:class:: mindspore.ops.ScatterSub(use_locking=False)

    使用给定更新值通过减法操作和输入索引来更新Tensor值。此操作在更新完成后输出数据 ，这有利于更加方便地使用更新后的值。

    对于每个在 `indices.shape` 中的 `i, ..., j` ：

    .. math::
        \text{input_x}[\text{indices}[i, ..., j], :] \mathrel{-}= \text{updates}[i, ..., j, :]

    输入的 `input_x` 和 `updates` 遵循隐式类型转换规则，以确保数据类型一致。如果它们具有不同的数据类型，则优先级低的数据类型将转换为优先级相对最高的数据类型。当需要转换Parameter的数据类型时，会抛出RuntimeError异常。

    参数：
        - **use_locking** (bool) - 表示是否使用锁来保护。默认值： ``False`` 。

    输入：
        - **input_x** (Parameter) - ScatterSub的输入，数据类型为Parameter。其shape为 :math:`(N, *)` ，其中 :math:`*` 为任意数量的额外维度。
        - **indices** (Tensor) - 指定相减操作的索引，其数据类型必须为mindspore.int32。
        - **updates** (Tensor) - 指定与 `input_x` 相减的Tensor，其数据类型与 `input_x` 的相同，shape为 `indices_shape + x_shape[1:]` 。

    输出：
        Tensor，表示更新后的 `input_x` ，其shape和数据类型与 `input_x` 的相同。

    异常：
        - **TypeError** - `use_locking` 不是bool。
        - **TypeError** - `indices` 不是int32。
        - **ValueError** - `updates` 的shape不是 `indices_shape + x_shape[1:]` 。
        - **RuntimeError** - 当 `input_x` 和 `updates` 类型不一致，需要进行类型转换时，如果 `updates` 不支持转成参数 `input_x` 需要的数据类型，就会报错。