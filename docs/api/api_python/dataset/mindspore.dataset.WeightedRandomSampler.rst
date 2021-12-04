mindspore.dataset.WeightedRandomSampler
=======================================

.. py:class:: mindspore.dataset.WeightedRandomSampler(weights, num_samples=None, replacement=True)

    使用给定的权重（概率）进行随机采样[0，len(weights) - 1]中的元素。

    **参数：**

    - **weights** (list[float, int]) - 权重序列，总和不一定为1。
    - **num_samples** (int, optional) - 待采样的元素数量（默认值为None，代表采样所有元素）。
    - **replacement** (bool) - 如果值为True，则将样本ID放回下一次采样（默认值为True）。

    **样例：**

    >>> weights = [0.9, 0.01, 0.4, 0.8, 0.1, 0.1, 0.3]
    >>>
    >>> # 创建一个WeightedRandomSampler，将对4个元素进行有放回采样
    >>> sampler = ds.WeightedRandomSampler(weights, 4)
    >>> dataset = ds.ImageFolderDataset(image_folder_dataset_dir,
    ...                                 num_parallel_workers=8,
    ...                                 sampler=sampler)

    **异常：**

    - **TypeError：** `weights` 元素的类型不是number。
    - **TypeError：** `num_samples` 不是整数值。
    - **TypeError：** `replacement` 不是布尔值。
    - **RuntimeError：** `weights` 为空或全为零。
    - **ValueError：** `num_samples` 为负值。

    .. include:: mindspore.dataset.BuiltinSampler.rst