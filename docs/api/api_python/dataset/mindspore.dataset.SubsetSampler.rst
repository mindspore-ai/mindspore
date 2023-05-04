mindspore.dataset.SubsetSampler
====================================

.. py:class:: mindspore.dataset.SubsetSampler(indices, num_samples=None)

    给定样本的索引序列，对数据集采样指定索引的样本。

    参数：
        - **indices** (Iterable) - 索引的序列（包括除了string类型的任意Python可迭代对象类型）。
        - **num_samples** (int, 可选) - 获取的样本数，可用于部分获取采样得到的样本。默认值： ``None`` ，获取采样到的所有样本。

    异常：
        - **TypeError** - `indices` 的类型不是int。
        - **TypeError** - `num_samples` 的类型不是int。
        - **ValueError** - `num_samples` 为负值。

    .. include:: mindspore.dataset.BuiltinSampler.rst
