mindspore.dataset.dataloader.default_collate
============================================

.. py:function:: mindspore.dataset.dataloader.default_collate(batch)

    当在 :class:`~mindspore.dataset.dataloader.DataLoader` 中启用批处理时，默认使用此函数将批数据沿第一个维度进行拼接。

    此函数使用预定义的从数据类型到整理函数的映射来执行以下类型转换，然后根据
    :func:`~mindspore.dataset.dataloader._utils.collate.collate` 中描述的规则整理输入批数据：

    * :py:class:`list` [:class:`mindspore.Tensor`] -> :class:`mindspore.Tensor`
    * :py:class:`list` [:class:`numpy.ndarray`] -> :class:`mindspore.Tensor`
    * :py:class:`list` [:py:class:`float`] -> :class:`mindspore.Tensor`
    * :py:class:`list` [:py:class:`int`] -> :class:`mindspore.Tensor`
    * :py:class:`list` [:py:class:`str`] -> :py:class:`list` [:py:class:`str`]
    * :py:class:`list` [:py:class:`bytes`] -> :py:class:`list` [:py:class:`bytes`]

    参数：
        - **batch** (list) - 要整理的批数据。

    返回：
        :py:class:`~typing.Any` ，整理后的数据。
