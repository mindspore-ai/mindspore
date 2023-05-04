mindspore.dataset.DistributedSampler
====================================

.. py:class:: mindspore.dataset.DistributedSampler(num_shards, shard_id, shuffle=True, num_samples=None, offset=-1)

    分布式采样器，将数据集进行分片用于分布式训练。

    参数：
        - **num_shards** (int) - 数据集分片数量。
        - **shard_id** (int) - 当前分片的分片ID，应在[0, num_shards-1]范围内。
        - **shuffle** (bool, 可选) - 是否混洗采样得到的样本。默认值： ``True`` ，混洗样本。
        - **num_samples** (int, 可选) - 获取的样本数，可用于部分获取采样得到的样本。默认值： ``None`` ，获取采样到的所有样本。
        - **offset** (int, 可选) - 分布式采样结果进行分配时的起始分片ID号，值不能大于参数 `num_shards` 。从不同的分片ID开始分配数据可能会影响每个分片的最终样本数。仅当ConcatDataset以 :class:`mindspore.dataset.DistributedSampler` 为采样器时，此参数才有效。默认值： ``-1`` ，每个分片具有相同的样本数。

    异常：
        - **TypeError** - `num_shards` 的类型不是int。
        - **TypeError** - `shard_id` 的类型不是int。
        - **TypeError** - `shuffle` 的类型不是bool。
        - **TypeError** - `num_samples` 的类型不是int。
        - **TypeError** - `offset` 的类型不是int。
        - **ValueError** - `num_samples` 为负值。
        - **RuntimeError** - `num_shards` 不是正值。
        - **RuntimeError** - `shard_id` 小于0或大于等于 `num_shards` 。
        - **RuntimeError** - `offset` 大于 `num_shards` 。

    .. include:: mindspore.dataset.BuiltinSampler.rst

    .. include:: mindspore.dataset.BuiltinSampler.b.rst