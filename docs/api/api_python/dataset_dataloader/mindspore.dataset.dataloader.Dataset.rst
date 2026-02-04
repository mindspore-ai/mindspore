mindspore.dataset.dataloader.Dataset
====================================

.. py:class:: mindspore.dataset.dataloader.Dataset()

    所有数据集的基类。

    Map style 数据集应该继承自此类。

    Map style 数据集表示了一种从键值到数据样本的映射。
    子类必须重写 :meth:`__getitem__` 方法，定义如何根据键值检索样本。
    子类可以选择性地重写 :meth:`__len__` 方法，返回数据集的样本总数。
    如果未实现，某些内置采样器和 :class:`~mindspore.dataset.dataloader.DataLoader` 方法将会不可用。
