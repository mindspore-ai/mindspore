mindspore.ops.scatter_mul
=========================

.. py:function:: mindspore.ops.scatter_mul(input_x, indices, updates)

    根据指定更新值和输入索引通过乘法运算更新输入数据的值。

    对于 `indices.shape` 的每个 `i, ..., j` ：

    .. math::
        \text{input_x}[\text{indices}[i, ..., j], :] \mathrel{*}= \text{updates}[i, ..., j, :]

    输入的 `input_x` 和 `updates` 遵循隐式类型转换规则，以确保数据类型一致。如果数据类型不同，则低精度数据类型将转换为高精度的数据类型。当参数的数据类型需要转换时，则会抛出RuntimeError异常。

    参数：
        - **input_x** (Parameter) - ScatterMul的输入，任意维度的Parameter。
        - **indices** (Tensor) - 指定相乘操作的索引，数据类型必须为mindspore.int32。
        - **updates** (Tensor) - 指定与 `input_x` 相乘的Tensor，数据类型与 `input_x` 相同，shape为 `indices.shape + input_x.shape[1:]` 。

    返回：
        Tensor，更新后的 `input_x` ，shape和类型与 `input_x` 相同。

    异常：
        - **TypeError** - `use_locking` 不是bool。
        - **TypeError** - `indices` 不是int32，也不是int64。
        - **ValueError** - `updates` 的shape不等于 `indices.shape + input_x.shape[1:]` 。
        - **RuntimeError** - 当 `input_x` 和 `updates` 类型不一致，需要进行类型转换时，如果 `updates` 不支持转成参数 `input_x` 需要的数据类型，就会报错。