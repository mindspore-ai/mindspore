mindspore.ops.Tile
===================

.. py:class:: mindspore.ops.Tile

    按照给定的次数复制输入Tensor。

    更多参考详见 :func:`mindspore.ops.tile`。

    输入：
        - **input_x** (Tensor) - 1-D或更高维的Tensor，设其shape为 :math:`(x_1, x_2, ..., x_S)` 。
        - **multiples** (tuple[int]) - 指定复制次数的参数，参数类型为tuple，数据类型为整数。如 :math:`(y_1, y_2, ..., y_S)` 。 `multiples` 的长度不能小于 `input_x` 的维度。只支持常量值。

    输出：
        Tensor，具有与 `input_x` 相同的数据类型。假设 `multiples` 的长度为 `d` ， `input_x` 的维度为 `input_x.dim` ， `input_x` 的shape为 :math:`(x_1, x_2, ..., x_S)` 。

        - 如果 `input_x.dim = d` ，将其相应位置的shape相乘，输出的shape为 :math:`(x_1*y_1, x_2*y_2, ..., x_S*y_S)` 。
        - 如果 `input_x.dim < d` ，在 `input_x` 的shape的前面填充1，直到它们的长度一致。例如将 `input_x` 的shape设置为 :math:`(1, ..., x_1, x_2, ..., x_S)` ，然后可以将其相应位置的shape相乘，输出的shape为 :math:`(1*y_1, ..., x_R*y_R, x_S*y_S)` 。
