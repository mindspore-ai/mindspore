mindspore.ops.ScatterMin
=========================

.. py:class:: mindspore.ops.ScatterMin(use_locking=False)

    通过最小操作更新输入张量的值。

    根据指定更新值和输入索引通过最小值操作更新输入数据的值。
    该操作在更新完成后输出 `input_x` ，这样方便使用更新后的值。

    对于 `indices.shape` 的每个 :math:`i, ..., j` ：

    .. math::
        \text{input_x}[\text{indices}[i, ..., j], :]
        = min(\text{input_x}[\text{indices}[i, ..., j], :], \text{updates}[i, ..., j, :])

    输入的 `input_x` 和 `updates` 遵循隐式类型转换规则，以确保数据类型一致。如果数据类型不同，则低精度数据类型将转换为高精度的数据类型。当 `updates` 不支持转成 `input_x` 需要的数据类型时，则会抛出RuntimeError异常。

    参数：
        - **use_locking** (bool) - 是否启用锁保护。默认值：False。

    输入：
        - **input_x** (Parameter) - ScatterMin的输入，任意维度的Parameter。shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。
        - **indices** (Tensor) - 指定最小值操作的索引，数据类型必须为mindspore.int32或者mindspore.int64。
        - **updates** (Tensor) - 指定与 `input_x` 取最小值操作的Tensor，数据类型与 `input_x` 相同，shape为 `indices.shape + x.shape[1:]` 。

    输出：
        Tensor，更新后的 `input_x` ，shape和类型与 `input_x` 相同。

    异常：
        - **TypeError** - `use_locking` 不是bool。
        - **TypeError** - `indices` 不是int32或者int64。
        - **ValueError** - `updates` 的shape不等于 `indices.shape + x.shape[1:]` 。
        - **RuntimeError** - 当 `input_x` 和 `updates` 类型不一致，需要进行类型转换时，如果 `updates` 不支持转成参数 `input_x` 需要的数据类型，就会报错。
        - **RuntimeError** - 在Ascend平台上，输入的 `input_x` ， `indices` 和 `updates` 的数据维度大于八维。