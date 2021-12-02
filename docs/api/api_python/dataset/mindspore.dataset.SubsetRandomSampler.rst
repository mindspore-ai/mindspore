mindspore.dataset.SubsetRandomSampler
======================================

.. py:class:: mindspore.dataset.SubsetRandomSampler(indices, num_samples=None)

    对索引序列中的元素进行随机采样。

    **参数：**

    - **indices** (Any iterable Python object but string): 索引的序列。
    - **num_samples** (int, optional): 要采样的元素数量（默认值为None，采样所有元素）。

    **样例：**

    >>> indices = [0, 1, 2, 3, 7, 88, 119]
    >>>
    >>> # 创建一个SubsetRandomSampler，从提供的索引中采样
    >>> sampler = ds.SubsetRandomSampler(indices)
    >>> data = ds.ImageFolderDataset(image_folder_dataset_dir, num_parallel_workers=8, sampler=sampler)

    **异常：**

    - **TypeError：** `indices` 的类型不是数字。
    - **TypeError：** `num_samples` 不是整数值。
    - **ValueError：** `num_samples` 为负值。

    .. include:: mindspore.dataset.BuiltinSampler.rst