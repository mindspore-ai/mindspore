mindspore.dataset.MindDataset
==============================

.. py:class:: mindspore.dataset.MindDataset(dataset_files, columns_list=None, num_parallel_workers=None, shuffle=None, num_shards=None, shard_id=None, sampler=None, padded_sample=None, num_padded=None, num_samples=None, cache=None)

    读取和解析MindRecord数据文件构建数据集。生成的数据集的列名和列类型取决于MindRecord文件中保存的列名与类型。

    参数：
        - **dataset_files** (Union[str, list[str]]) - MindRecord文件路径，支持单文件路径字符串、多文件路径字符串列表。如果 `dataset_files` 的类型是字符串，则它代表一组具有相同前缀名的MindRecord文件，同一路径下具有相同前缀名的其他MindRecord文件将会被自动寻找并加载。如果 `dataset_files` 的类型是列表，则它表示所需读取的MindRecord数据文件列表。
        - **columns_list** (list[str]，可选) - 指定从MindRecord文件中读取的数据列。默认值： ``None`` ，读取所有列。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值： ``None`` ，使用全局默认线程数（8），也可以通过 :func:`mindspore.dataset.config.set_num_parallel_workers` 配置全局线程数。
        - **shuffle** (Union[bool, :class:`~.dataset.Shuffle`], 可选) - 每个epoch中数据混洗的模式，支持传入bool类型与枚举类型进行指定。默认值： ``None`` ，采用 ``Shuffle.ADAPTIVE`` 。
          如果 `shuffle` 为 ``False`` ，则不混洗；如果 `shuffle` 为 ``True`` ，等同于将 `shuffle` 设置为 ``Shuffle.ADAPTIVE`` 。
          通过传入枚举变量设置数据混洗的模式，枚举变量参考链接 :class:`mindspore.dataset.Shuffle` ：

          - ``Shuffle.ADAPTIVE`` ：当数据集样本小于等于1亿时，采用 ``Shuffle.GLOBAL`` ；当大于1亿时，采用局部 ``Shuffle.PARTIAL`` ，每100万样本混洗一次。
          - ``Shuffle.GLOBAL`` ：执行全局混洗，一次性混洗数据集中所有样本。占用内存大。
          - ``Shuffle.PARTIAL`` ：执行局部混洗，每100万个样本混洗一次。占用内存小于 ``Shuffle.GLOBAL`` 。
          - ``Shuffle.FILES`` ：仅混洗文件序列，不混洗文件中的数据。
          - ``Shuffle.INFILE`` ：保持读入文件的序列，仅混洗每个文件中的数据。

        - **num_shards** (int, 可选) - 指定分布式训练时数据集划分的片数。默认值： ``None`` 。指定此参数后， `num_samples` 表示每个分片的最大样本数。一般在 `数据并行模式训练 <https://www.mindspore.cn/tutorials/zh-CN/master/parallel/data_parallel.html#数据集加载>`_ 的时候使用。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值： ``None`` 。只有当指定了 `num_shards` 时才能指定此参数。
        - **sampler** (Sampler, 可选) - 指定从数据集中选取样本的采样器。默认值： ``None`` 。下表中会展示不同配置的预期行为。当前此数据集仅支持以下采样器： :class:`mindspore.dataset.SubsetRandomSampler` 、 :class:`mindspore.dataset.PKSampler` 、 :class:`mindspore.dataset.RandomSampler` 、 :class:`mindspore.dataset.SequentialSampler` 和 :class:`mindspore.dataset.DistributedSampler` 。
        - **padded_sample** (dict, 可选) - 指定额外添加到数据集的样本，可用于在分布式训练时补齐分片数据，注意字典的键名需要与 `columns_list` 指定的列名相同。默认值： ``None`` ，不添加样本。需要与 `num_padded` 参数同时使用。
        - **num_padded** (int, 可选) - 指定额外添加的数据集样本的数量。在分布式训练时可用于为数据集补齐样本，使得总样本数量可被 `num_shards` 整除。默认值： ``None`` ，不添加样本。需要与 `padded_sample` 参数同时使用。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值： ``None`` ，读取所有样本。
        - **cache** (:class:`~.dataset.DatasetCache`, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/zh-CN/master/dataset/cache.html>`_ 。默认值： ``None`` ，不使用缓存。

    异常：
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - `dataset_files` 参数所指向的文件无效或不存在。
        - **ValueError** - `num_parallel_workers` 参数超过最大线程数。
        - **ValueError** - 如果 `shard_id` 取值不在[0, `num_shards` )范围。
        - **TypeError** - `shuffle` 的类型不是None 或者 bool 或者 Shuffle。

    教程样例：
        - `使用数据Pipeline加载 & 处理数据集
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/dataset_gallery.html>`_

    .. note::
        对MindRecord进行分片（配置 `num_shards` 和 `shard_id` ）时，数据的切分逻辑有2种实现策略。此API默认采用策略1，可通过设置环境变量 `MS_DEV_MINDRECORD_SHARD_BY_BLOCK=True` 切换回策略2。该环境变量只对 `DistributedSampler` 采样器生效。

        .. list-table:: 数据分片的实现策略1
            :widths: 50 50 50 50
            :header-rows: 1

            * - rank 0
              - rank 1
              - rank 2
              - rank 3
            * - 0
              - 1
              - 2
              - 3
            * - 4
              - 5
              - 6
              - 7
            * - 8
              - 9
              - 10
              - 11

        .. list-table:: 数据分片的实现策略2
            :widths: 50 50 50 50
            :header-rows: 1

            * - rank 0
              - rank 1
              - rank 2
              - rank 3
            * - 0
              - 3
              - 6
              - 9
            * - 1
              - 4
              - 7
              - 10
            * - 2
              - 5
              - 8
              - 11

    .. note:: 入参 `num_samples` 、 `shuffle` 、 `num_shards` 、 `shard_id` 可用于控制数据集所使用的采样器，其与入参 `sampler` 搭配使用的效果如下。

    .. include:: mindspore.dataset.sampler.rst
        :parser: reStructuredText

.. include:: mindspore.dataset.api_list_nlp.rst
