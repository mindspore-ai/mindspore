mindspore.dataset.SubsetRandomSampler
======================================

.. py:class:: mindspore.dataset.SubsetRandomSampler(indices, num_samples=None)

    给定样本的索引序列，从序列中随机获取索引对数据集进行采样。

    **参数：**

    - **indices** (Iterable): 样本索引的序列（除了string类型外的任意Python可迭代对象类型）。
    - **num_samples** (int, 可选): 获取的样本数，可用于部分获取采样得到的样本。默认值：None，获取采样到的所有样本。

    **异常：**

    - **TypeError：** `indices` 的类型不是整数。
    - **TypeError：** `num_samples` 不是整数值。
    - **ValueError：** `num_samples` 为负值。

    **样例：**

    >>> indices = [0, 1, 2, 3, 7, 88, 119]
    >>>
    >>> # 创建一个SubsetRandomSampler，根据提供的索引序列，对数据集进行随机采样
    >>> sampler = ds.SubsetRandomSampler(indices)
    >>> data = ds.ImageFolderDataset(image_folder_dataset_dir, num_parallel_workers=8, sampler=sampler)

    .. include:: mindspore.dataset.BuiltinSampler.rst