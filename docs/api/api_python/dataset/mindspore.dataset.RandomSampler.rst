mindspore.dataset.RandomSampler
================================

.. py:class:: mindspore.dataset.RandomSampler(replacement=False, num_samples=None)

    随机采样器。

    **参数：**

    - **replacement** (bool, optional): 如果为True，则将样本ID放回下一次采样（默认值为False）。
    - **num_samples** (int, optional): 要采样的元素数量（默认值为None，采样所有元素）。

    **样例：**

    >>> # 创建一个RandomSampler
    >>> sampler = ds.RandomSampler()
    >>> dataset = ds.ImageFolderDataset(image_folder_dataset_dir,
    ...                                 num_parallel_workers=8,
    ...                                 sampler=sampler)

    **异常：**

    - **TypeError：** `replacement` 不是bool值。
    - **TypeError：** `num_samples` 不是整数值。
    - **ValueError：** `num_samples` 为负值。

    .. include:: mindspore.dataset.BuiltinSampler.rst