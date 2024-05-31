mindspore.mint.tile
===================

.. py:function:: mindspore.mint.tile(input, dims)

    通过复制 `dims` 次 `input` 来创建新的Tensor。输出Tensor的第i维度有 `input.shape[i] * dims[i]` 个元素，并且 `input` 的值沿第i维度被复制 `dims[i]` 次。

    参数：
        - **input** (Tensor) - 需要被复制的Tensor，shape为 :math:`(x_1, x_2, ..., x_S)` 。
        - **dims** (tuple[int]) - 指定复制次数的参数，参数类型为tuple，数据类型为整数。如 :math:`(y_1, y_2, ..., y_S)` 。 只支持常量值。

    返回：
        Tensor，具有与 `input` 相同的数据类型。假设 `dims` 的长度为 `d` ，`input` 的维度为 `input.dim` ，`input` 的shape为 :math:`(x_1, x_2, ..., x_S)` 。

        - 如果 `input.dim = d` ，将其相应位置的shape相乘，输出的shape为 :math:`(x_1*y_1, x_2*y_2, ..., x_S*y_S)` 。
        - 如果 `input.dim < d` ，在 `input` 的shape的前面填充1，直到它们的长度一致。例如将 `input` 的shape设置为 :math:`(1, ..., x_1, x_2, ..., x_S)` ，然后可以将其相应位置的shape相乘，输出的shape为 :math:`(1*y_1, ..., x_R*y_R, x_S*y_S)` 。
        - 如果 `input.dim > d` ，在 `dims` 的前面填充1，直到它们的长度一致。例如将 `dims` 设置为 :math:`(1, ..., y_1, y_2, ..., y_S)` ，然后可以将其相应位置的shape相乘，输出的shape为 :math:`(x_1*1, ..., x_R*y_R, x_S*y_S)` 。

    异常：
        - **TypeError** - `dims` 不是tuple或者其元素并非全部是int。
        - **ValueError** - `dims` 的元素并非全部大于或等于0。
