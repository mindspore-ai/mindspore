mindspore.Tensor.tile
=====================

.. py:method:: mindspore.Tensor.tile(dims) -> Tensor

    按照给定的次数复制输入Tensor。

    .. note::
        在Ascend平台上， `dims` 参数的个数不大于8，当前不支持超过4个维度同时做repeat的场景。

    参数：
         - **dims** (tuple[int]) - 指定复制次数的参数，如 :math:`(y_1, y_2, ..., y_S)` 。只支持常量值。

    返回：
        Tensor，具有与 `self` 相同的数据类型。假设 `dims` 的长度为 `d` ， `self` 的维度为 `self.dim` ， `self` 的shape为 :math:`(x_1, x_2, ..., x_S)` 。

        - 如果 `self.dim = d` ，将其相应位置的shape相乘，输出的shape为 :math:`(x_1*y_1, x_2*y_2, ..., x_S*y_S)` 。
        - 如果 `self.dim < d` ，在 `self` 的shape的前面填充1，直到它们的长度一致。例如将 `self` 的shape设置为 :math:`(1, ..., x_1, x_2, ..., x_S)` ，然后可以将其相应位置的shape相乘，输出的shape为 :math:`(1*y_1, ..., x_R*y_R, x_S*y_S)` 。
        - 如果 `self.dim > d` ，在 `dims` 的前面填充1，直到它们的长度一致。例如将 `dims` 设置为 :math:`(1, ..., y_1, y_2, ..., y_S)` ，然后可以将其相应位置的shape相乘，输出的shape为 :math:`(x_1*1, ..., x_R*y_R, x_S*y_S)` 。

    异常：
        - **TypeError** - `dims` 不是tuple或者其元素并非全部是int。
        - **ValueError** - `dims` 的元素并非全部大于或等于0。

    .. py:method:: mindspore.Tensor.tile(reps) -> Tensor
        :noindex:

    详情请参考 :func:`mindspore.ops.tile`，当前接口的参数 `reps` 和详情参考接口中的参数 `dims` 实际为同一参数。
