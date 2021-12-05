mindspore.dataset.SequentialSampler
===================================

.. py:class:: mindspore.dataset.SequentialSampler(start_index=None, num_samples=None)

    按顺序采样数据集元素，相当于不使用采样器。

    **参数：**

    - **start_index** (int, optional): 开始采样的索引。（默认值为None，从第一个ID开始）
    - **num_samples** (int, optional): 要采样的元素数量。（默认值为None，采样所有元素）

    **样例：**

    >>> # 创建SequentialSampler
    >>> sampler = ds.SequentialSampler()
    >>> dataset = ds.ImageFolderDataset(image_folder_dataset_dir,
    ...                                 num_parallel_workers=8,
    ...                                 sampler=sampler)

    **异常：**

    - **TypeError：** `start_index` 不是整数值。
    - **TypeError：** `num_samples` 不是整数值。
    - **RuntimeError：** `start_index` 为负值。
    - **ValueError：** `num_samples` 为负值。

    .. include:: mindspore.dataset.BuiltinSampler.rst