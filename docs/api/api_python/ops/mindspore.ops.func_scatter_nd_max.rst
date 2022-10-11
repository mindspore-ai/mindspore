mindspore.ops.scatter_nd_max
============================

.. py:function:: mindspore.ops.scatter_nd_max(input_x, indices, updates, use_locking=False)

    对张量中的单个值或切片应用sparse maximum。

    使用给定值通过最大值运算和输入索引更新Parameter值。在更新完成后输出 `input_x` ，这有利于更加方便地使用更新后的值。

    `input_x` 的rank为P， `indices` 的rank为Q， `Q >= 2` 。

    `indices` 的shape为 :math:`(i_0, i_1, ..., i_{Q-2}, N)` ， `N <= P` 。

    `indices` 的最后一个维度（长度为 `N` ）表示沿着 `input_x` 的 `N` 个维度进行切片。

    `updates` 表示rank为 `Q-1+P-N` 的Tensor，shape为 :math:`(i_0, i_1, ..., i_{Q-2}, x\_shape_N, ..., x\_shape_{P-1})` 。

    参数：
        - **input_x** (Parameter) - 输入参数，任意维度的Parameter。
        - **indices** (Tensor) - 指定最大值操作的索引，数据类型为mindspore.int32或mindspore.int64。索引的rank必须至少为2，并且 `indices.shape[-1] <= len(shape)` 。
        - **updates** (Tensor) - 指定与 `input_x` 操作的Tensor，数据类型与 `input_x` 相同，shape为 `indices.shape[:-1] + x.shape[indices.shape[-1]:]` 。
        - **use_locking** (bool) - 是否启用锁保护。默认值：False。

    返回：
        Tensor，更新后的 `input_x` ，shape和数据类型与 `input_x` 相同。

    异常：
        - **TypeError** - `use_locking` 的数据类型不是bool。
        - **TypeError** - `indices` 的数据类型不是int32或int64。
        - **TypeError** - `input_x` 和 `updates` 的数据类型不相同。
        - **ValueError** - `updates` 的shape不等于 `indices.shape[:-1] + x.shape[indices.shape[-1]:]` 。
        - **RuntimeError** - 当 `input_x` 和 `updates` 类型不一致，需要进行类型转换时，如果 `updates` 不支持转成参数 `input_x` 需要的数据类型，就会报错。
