mindspore.dataset.SequentialSampler
===================================

.. py:class:: mindspore.dataset.SequentialSampler(start_index=None, num_samples=None)

    按数据集的读取顺序采样数据集样本，相当于不使用采样器。

    参数：
        - **start_index** (int, 可选) - 采样的起始样本ID。默认值：None，从数据集第一个样本开始采样。
        - **num_samples** (int, 可选) - 获取的样本数，可用于部分获取采样得到的样本。默认值：None，获取采样到的所有样本。

    异常：
        - **TypeError** - `start_index` 的类型不是int。
        - **TypeError** - `num_samples` 的类型不是int。
        - **RuntimeError** - `start_index` 为负值。
        - **ValueError** - `num_samples` 为负值。

    .. include:: mindspore.dataset.BuiltinSampler.rst

    .. include:: mindspore.dataset.BuiltinSampler.b.rst