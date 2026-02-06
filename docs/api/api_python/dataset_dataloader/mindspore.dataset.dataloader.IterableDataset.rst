mindspore.dataset.dataloader.IterableDataset
============================================

.. py:class:: mindspore.dataset.dataloader.IterableDataset()

    可迭代数据集的基类。

    Iterable style 数据集应该继承自此类。

    Iterable style 数据集表示了一种返回数据样本的可迭代对象。当随机读取数据的成本高昂甚至无法实现时尤其有用。
    子类必须重写 :meth:`__iter__` 方法，返回用于迭代数据集样本的迭代器。
