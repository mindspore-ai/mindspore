Class mindspore.dataset.WeightedRandomSampler(weights, num_samples=None, replacement=True)

    使用给定的权重（概率）进行随机采样[0，len(weights) - 1]中的元素。

    参数：
        weights (list[float, int])：权重序列，总和不一定为1。
        num_samples (int, optional)：待采样的元素数量（默认值为None，代表采样所有元素）。
        replacement (bool)：如果值为True，则将样本ID放回下一次采样（默认值为True）。

    示例：
        >>> weights = [0.9, 0.01, 0.4, 0.8, 0.1, 0.1, 0.3]
        >>>
        >>> # creates a WeightedRandomSampler that will sample 4 elements without replacement
        >>> sampler = ds.WeightedRandomSampler(weights, 4)
        >>> dataset = ds.ImageFolderDataset(image_folder_dataset_dir,
        ...                                 num_parallel_workers=8,
        ...                                 sampler=sampler)

    异常：
        TypeError：weights元素的类型不是number。
        TypeError：num_samples不是整数值。
        TypeError：replacement不是布尔值。
        RuntimeError：weights为空或全为零。
        ValueError：num_samples为负值。

    .. include:: mindspore.dataset.BuiltinSampler.rst