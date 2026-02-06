mindspore.dataset.DistributedSampler
====================================

.. py:class:: mindspore.dataset.DistributedSampler(num_shards, shard_id, shuffle=True, num_samples=None, offset=-1)

    分布式采样器，将数据集进行分片用于分布式训练。

    .. note::
        不同数据集的混洗模式支持如下：

        .. list-table:: 混洗模式支持
            :widths: 50 50 50 50
            :header-rows: 1

            * - 混洗模式
              - MindDataset
              - TFRecordDataset
              - 其他数据集
            * - ``Shuffle.ADAPTIVE``
              - 支持
              - 不支持
              - 不支持
            * - ``Shuffle.GLOBAL``
              - 支持
              - 支持
              - 支持
            * - ``Shuffle.PARTIAL``
              - 支持
              - 不支持
              - 不支持
            * - ``Shuffle.FILES``
              - 支持
              - 支持
              - 不支持
            * - ``Shuffle.INFILE``
              - 支持
              - 不支持
              - 不支持

    参数：
        - **num_shards** (int) - 数据集分片数量。
        - **shard_id** (int) - 当前分片的分片ID，应在[0, num_shards-1]范围内。
        - **shuffle** (Union[bool, Shuffle], 可选) - 是否混洗采样得到的样本。默认值： ``True`` ，采用 ``mindspore.dataset.Shuffle.GLOBAL`` 混洗样本。如果 `shuffle` 为 ``False`` ，则不混洗。
          通过传入枚举变量设置数据混洗的模式，枚举变量参考链接 :class:`mindspore.dataset.Shuffle` ：

          - ``Shuffle.ADAPTIVE`` ：当数据集样本小于等于1亿时，采用 ``Shuffle.GLOBAL`` ；当大于1亿时，采用局部 ``Shuffle.PARTIAL`` ，每100万样本混洗一次。
          - ``Shuffle.GLOBAL`` ：执行全局混洗，一次性混洗数据集中所有样本。
          - ``Shuffle.PARTIAL`` ：执行局部混洗，每100万个样本混洗一次。
          - ``Shuffle.FILES`` ：仅混洗文件序列，不混洗文件中的数据。
          - ``Shuffle.INFILE`` ：保持读入文件的序列，仅混洗每个文件中的数据。

        - **num_samples** (int, 可选) - 获取的样本数，可用于获取部分采样得到的样本。默认值： ``None`` ，获取采样到的所有样本。
        - **offset** (int, 可选) - 分布式采样结果进行分配时的起始分片ID号，值不能大于参数 `num_shards` 。从不同的分片ID开始分配数据可能会影响每个分片的最终样本数。仅当ConcatDataset以 :class:`mindspore.dataset.DistributedSampler` 为采样器时，此参数才有效。默认值： ``-1`` ，表示每个分片具有相同的样本数。

    异常：
        - **TypeError** - `num_shards` 的类型不是int。
        - **TypeError** - `shard_id` 的类型不是int。
        - **TypeError** - `shuffle` 的类型不是bool 或者 Shuffle。
        - **TypeError** - `num_samples` 的类型不是int。
        - **TypeError** - `offset` 的类型不是int。
        - **ValueError** - `num_samples` 为负值。
        - **RuntimeError** - `num_shards` 不是正值。
        - **RuntimeError** - `shard_id` 小于0或大于等于 `num_shards` 。
        - **RuntimeError** - `offset` 大于 `num_shards` 。

    .. include:: mindspore.dataset.BuiltinSampler.rst

    .. include:: mindspore.dataset.BuiltinSampler.b.rst
