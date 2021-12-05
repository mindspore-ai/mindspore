mindspore.dataset.DistributedSampler
====================================

.. py:class:: mindspore.dataset.DistributedSampler(num_shards, shard_id, shuffle=True, num_samples=None, offset=-1)

    分布式采样器，将数据集进行分片用于分布式训练。

    **参数：**

    - **num_shards** (int) - 数据集分片数量。
    - **shard_id** (int) - 当前分片的分片ID，应在[0, num_shards-1]范围内。
    - **shuffle** (bool, optional) - 如果为True，则索引将被打乱（默认为True）。
    - **num_samples** (int, optional) - 要采样的样本数（默认为None，对所有元素进行采样）。
    - **offset** (int, optional) - 将数据集中的元素发送到的起始分片ID，不应超过 `num_shards` 。仅当ConcatDataset以DistributedSampler为采样器时，此参数才有效。此参数影响每个分片的样本数（默认为-1，每个分片具有相同的样本数）。

    **样例：**

    >>> # 创建一个分布式采样器，共10个分片。当前分片为分片5。
    >>> sampler = ds.DistributedSampler(10, 5)
    >>> dataset = ds.ImageFolderDataset(image_folder_dataset_dir,
    ...                                 num_parallel_workers=8,
    ...                                 sampler=sampler)

    **异常：**

    - **TypeError** - `num_shards` 不是整数值。
    - **TypeError** - `shard_id` 不是整数值。
    - **TypeError** - `shuffle` 不是Boolean值。
    - **TypeError** - `num_samples` 不是整数值。
    - **TypeError** - `offset` 不是整数值。
    - **ValueError** - `num_samples` 为负值。
    - **RuntimeError** - `num_shards` 不是正值。
    - **RuntimeError** - `shard_id` 小于0或大于等于 `num_shards` 。
    - **RuntimeError** - `offset` 大于 `num_shards` 。

    .. include:: mindspore.dataset.BuiltinSampler.rst