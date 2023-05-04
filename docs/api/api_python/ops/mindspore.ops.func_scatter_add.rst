mindspore.ops.scatter_add
=========================

.. py:function:: mindspore.ops.scatter_add(input_x, indices, updates)

    根据指定的更新值和输入索引，进行加法运算更新输入Tensor的值，返回更新后的Tensor。当同一索引有不同更新值时，更新的结果将是累积的加法的结果。

    参数：
        - **input_x** (Parameter) - scatter_add的输入，数据类型为Parameter。
        - **indices** (Tensor) - 指定加法操作的索引，数据类型必须为mindspore.int32或者mindspore.int64。
        - **updates** (Tensor) - 指定与 `input_x` 加法操作的Tensor，数据类型与 `input_x` 相同，shape为 `indices.shape + input_x.shape[1:]` 。

    返回：
        Tensor，更新后的 `input_x` ，shape和类型与 `input_x` 相同。

    异常：
        - **TypeError** - `indices` 不是int32或者int64。
        - **ValueError** - `updates` 的shape不等于 `indices.shape + input_x.shape[1:]` 。
        - **RuntimeError** - 当 `input_x` 和 `updates` 类型不一致，需要进行类型转换时，如果 `updates` 不支持转成参数 `input_x` 需要的数据类型，就会报错。
