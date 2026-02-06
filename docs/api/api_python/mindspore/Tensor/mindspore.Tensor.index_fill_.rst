mindspore.Tensor.index_fill\_
=============================

.. py:method:: mindspore.Tensor.index_fill_(dim, index, value) -> Tensor

    按 `index` 中给定的顺序选择索引，将输入 `value` 的值填充到 `self` Tensor的 `dim` 维的所有元素。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    .. note::
        计算 `value` 的梯度时， `index` 的值必须在 :math:`[0, self.shape[dim])` 范围内，如果超出该范围，结果未定义。

    参数：
        - **dim** (int) - 填充 `self` Tensor的维度。
        - **index** (Tensor) - 填充 `self` Tensor的索引。 `index` 必须是一个0D或1D Tensor，数据类型为int32或int64。
        - **value** (Union[Tensor, Number, bool]) - 填充 `self` Tensor的值。 `value` 为数值型、bool，或者数据类型为数值型或bool的Tensor。如果 `value` 是Tensor时，必须为0D Tensor。

    返回：
        Tensor，shape与 `self` 的shape相同，数据类型和 `self` 的数据类型相同。

    异常：
        - **TypeError** - 如果 `index` 的数据类型不是int32或int64。
        - **RuntimeError** - 如果 `dim` 不在 :math:`[-self.ndim, self.ndim)` 范围内。
        - **RuntimeError** - 如果 `index` 的维数大于1。
        - **RuntimeError** - 如果 `value` 是一个Tensor时，其维数不为0。
