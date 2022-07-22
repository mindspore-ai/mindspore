mindspore.ops.ScatterNonAliasingAdd
=====================================

.. py:class:: mindspore.ops.ScatterNonAliasingAdd

    使用给定值通过加法操作和输入索引来更新Tensor值。此操作在更新完成后输出 `input_x` ，这有利于更加方便地使用更新后的值。

    输入的 `input_x` 和 `updates` 遵循隐式类型转换规则，以确保数据类型一致。如果它们具有不同的数据类型，则低精度数据类型将转换为高精度数据类型。当需要转换Parameter的数据类型时，则会抛出RuntimeError异常。

    输入：
        - **input_x** (Parameter) - ScatterNonAliasingAdd的输入，任意维度的Parameter。其数据类型为float16、float32或int32。
        - **indices** (Tensor) - 指定加法操作的索引。其数据类型为mindspore.int32。
        - **updates** (Tensor) - 指定与 `input_x` 相加操作的Tensor。其数据类型与 `input_x` 相同，shape为 `indices.shape[:-1] + x.shape[indices.shape[-1]:]` 。

    输出：
        Parameter，表示更新后的 `input_x` 。

    异常：
        - **TypeError** - `indices` 的数据类型不是int32。
        - **TypeError** - `input_x` 的数据类型不是float16、float32或int32。
        - **ValueError** - `updates` 的shape不是 `indices.shape[:-1] + x.shape[indices.shape[-1]:]` 。
        - **RuntimeError** - 当 `input_x` 和 `updates` 类型不一致，需要进行类型转换时，如果 `updates` 不支持转成参数 `input_x` 需要的数据类型，就会报错。