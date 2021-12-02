mindspore.dataset.PKSampler
==============================

.. py:class:: mindspore.dataset.PKSampler(num_val, num_class=None, shuffle=False, class_column='label', num_samples=None)

    为数据集中的每个P类采样K个元素。

    **参数：**

    - **num_val** (int): 每个类要采样的元素数量。
    - **num_class** (int, optional): 要采样的类数量（默认值为None，采样所有类）。当前不支持指定该参数。
    - **shuffle** (bool, optional): 如果为True，则class ID将被打乱，否则它将不会被打乱（默认值为False）。
    - **class_column** (str, optional): 具有MindDataset类标签的列的名称（默认值'label'）。
    - **num_samples** (int, optional): 要采样的样本数（默认值为None，对所有元素进行采样）。

    **样例：**

    >>> # 创建一个PKSampler，从每个类中获取3个样本。
    >>> sampler = ds.PKSampler(3)
    >>> dataset = ds.ImageFolderDataset(image_folder_dataset_dir,
    ...                                 num_parallel_workers=8,
    ...                                 sampler=sampler)

    **异常：**

    - **TypeError：** `shuffle` 不是bool值。
    - **TypeError：** `class_column` 不是str值。
    - **TypeError：** `num_samples` 不是整数值。
    - **NotImplementedError：** `num_class` 不为None。
    - **RuntimeError：** `num_val` 不是正值。
    - **ValueError：** `num_samples` 为负值。

    .. include:: mindspore.dataset.BuiltinSampler.rst