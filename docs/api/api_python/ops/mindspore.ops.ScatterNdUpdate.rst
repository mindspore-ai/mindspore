mindspore.ops.ScatterNdUpdate
==============================

.. py:class:: mindspore.ops.ScatterNdUpdate(use_locking=True)

    使用给定值以及输入索引更新输入数据的值。

    `input_x` 的rank为P，而 `indices` 的rank为Q， `Q >= 2` 。

    `indices` 的shape为 :math:`(i_0, i_1, ..., i_{Q-2}, N)` ， `N <= P` 。

    `indices` 的最后一个维度（长度为 `N` ）表示沿着 `input_x` 的 `N` 个维度进行切片。

    `updates` 表示rank为 `Q-1+P-N` 的Tensor，shape为 :math:`(i_0, i_1, ..., i_{Q-2}, x\_shape_N, ..., x\_shape_{P-1})` 。

    输入的 `input_x` 和 `updates` 遵循隐式类型转换规则，以确保数据类型一致。如果数据类型不同，则低优先级数据类型将转换为相对最高优先级的数据类型。当需要参数的数据类型转换时，会抛出RuntimeError异常。

    参数：
        - **use_locking** (bool) - 是否启用锁保护。默认值：True。

    输入：
        - **input_x** (Parameter) - ScatterNdUpdate的输入，任意维度的Parameter。
        - **indices** (Tensor) - 指定更新操作的索引，数据类型为int32或者int64。
        - **updates** (Tensor) - 指定与 `input_x` 更新操作的Tensor，类型与输入相同。shape为 `indices.shape[:-1] + x.shape[indices.shape[-1]:]` 。

    输出：
        Tensor，shape和数据类型与输入 `input_x` 相同。

    异常：
        - **TypeError** - `use_locking` 不是bool。
        - **TypeError** - `indices` 不是int32或者int64。
        - **RuntimeError** - 当 `input_x` 和 `updates` 类型不一致，需要进行类型转换时，如果 `updates` 不支持转成参数 `input_x` 需要的数据类型，就会报错。