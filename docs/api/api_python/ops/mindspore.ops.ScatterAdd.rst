mindspore.ops.ScatterAdd
=========================

.. py:class:: mindspore.ops.ScatterAdd(use_locking=False)

    根据指定更新值和输入索引通过加法运算更新输入数据的值。

    对于 `indices.shape` 的每个 `i, ..., j` ：

    .. math::
        \text{input_x}[\text{indices}[i, ..., j], :] \mathrel{+}= \text{updates}[i, ..., j, :]

    输入的 `input_x` 和 `updates` 遵循隐式类型转换规则，以确保数据类型一致。如果数据类型不同，则低精度数据类型将转换为高精度的数据类型。当参数的数据类型需要转换时，则会抛出RuntimeError异常。

    .. note::
        这是一个运行即更新的算子。因此， `input_x` 在运算完成后即更新。

    参数：
        - **use_locking** (bool) - 是否启用锁保护。如果为 ``True`` ，则 `input_x` 将受到锁的保护。否则计算结果是未定义的。默认值： ``False`` 。

    输入：
        - **input_x** (Parameter) - ScatterAdd的输入，数据类型为Parameter。
        - **indices** (Tensor) - 指定相加操作的索引，数据类型为mindspore.int32或者mindspore.int64。
        - **updates** (Tensor) - 指定与 `input_x` 相加操作的Tensor，数据类型与 `input_x` 相同，shape为 `indices.shape + x.shape[1:]` 。

    输出：
        Tensor，更新后的 `input_x` ，shape和数据类型与 `input_x` 相同。

    异常：
        - **TypeError** - `use_locking` 不是bool。
        - **TypeError** - `indices` 不是int32或者int64。
        - **ValueError** - `updates` 的shape不等于 `indices.shape + x.shape[1:]` 。
        - **RuntimeError** - 当 `input_x` 和 `updates` 类型不一致，需要进行类型转换时，如果 `updates` 不支持转成参数 `input_x` 需要的数据类型，就会报错。
