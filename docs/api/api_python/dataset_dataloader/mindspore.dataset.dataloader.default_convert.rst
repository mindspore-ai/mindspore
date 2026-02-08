mindspore.dataset.dataloader.default_convert
============================================

.. py:function:: mindspore.dataset.dataloader.default_convert(data)

    当在 :class:`~mindspore.dataset.dataloader.DataLoader` 中禁用批处理时，默认使用此函数将NumPy数组类型的元素转换为
    :class:`mindspore.Tensor` 。

    * 如果输入是NumPy数组且其数据类型不为 `str`、`bytes` 或 `object`，则将其转换为 :class:`mindspore.Tensor`；
    * 如果输入是NumPy数值或布尔类型标量，则将其转换为 :class:`mindspore.Tensor`；
    * 如果输入是映射（ :py:class:`~collections.abc.Mapping` ）类型，则保持所有键不变，递归调用此函数转换各个键对应的值；
    * 如果输入是序列（ :py:class:`~collections.abc.Sequence` ）类型，则递归调用此函数转换各个位置的元素；
    * 否则，保持不变。

    参数：
        - **data** (:py:class:`~typing.Any`) - 要转换的单个数据。

    返回：
        :py:class:`~typing.Any` ，转换后的数据。
