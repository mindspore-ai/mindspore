mindspore.ops.tile
===================

.. py:function:: mindspore.ops.tile(input_x, reps)

    按照给定的次数复制输入Tensor。

    通过复制 `reps` 次 `input_x` 来创建新的Tensor。输出Tensor的第i维度有 `input_x.shape[i] * reps[i]` 个元素，并且 `input_x` 的值沿第i维度被复制 `multiples[i]` 次。

    .. note::
        `reps` 的长度必须大于或等于 `input_x` 的维度。

    参数：
        - **input_x** (Tensor) - 1-D或更高维的Tensor。
        - **reps** (tuple[int]) - 指定复制次数的参数，参数类型为tuple，数据类型为整数。如 :math:`(y_1, y_2, ..., y_S)` 。 `reps` 的长度不能小于 `input_x` 的维度。只支持常量值。

    返回：
        Tensor，具有与 `input_x` 相同的数据类型。假设 `reps` 的长度为 `d` ，`input_x` 的维度为 `input_x.dim` ，`input_x` 的shape为 :math:`(x_1, x_2, ..., x_S)` 。

        - 如果 `input_x.dim = d` ，将其相应位置的shape相乘，输出的shape为 :math:`(x_1*y_1, x_2*y_2, ..., x_S*y_S)` 。
        - 如果 `input_x.dim < d` ，在 `input_x` 的shape的前面填充1，直到它们的长度一致。例如将 `input_x` 的shape设置为 :math:`(1, ..., x_1, ..., x_R, x_S)` ，然后可以将其相应位置的shape相乘，输出的shape为 :math:`(1*y_1, ..., x_R*y_R, x_S*y_S)` 。

    异常：
        - **TypeError** - `reps` 不是tuple或者其元素并非全部是int。
        - **ValueError** - `reps` 的元素并非全部大于0。
        - **ValueError** - `reps` 的长度小于 `input_x` 中的维度。
