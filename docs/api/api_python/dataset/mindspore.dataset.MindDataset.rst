mindspore.dataset.MindDataset
==============================

.. py:class:: mindspore.dataset.MindDataset(dataset_files, columns_list=None, num_parallel_workers=None, shuffle=None, num_shards=None, shard_id=None, sampler=None, padded_sample=None, num_padded=None, num_samples=None, cache=None)

    读取和解析MindRecord数据文件构建数据集。生成的数据集的列名和列类型取决于MindRecord文件中的保存的列名与类型。

    参数：
        - **dataset_files** (Union[str, list[str]]) - MindRecord文件路径，支持单文件路径字符串、多文件路径字符串列表。如果 `dataset_files` 的类型是字符串，则它代表一组具有相同前缀名的MindRecord文件，同一路径下具有相同前缀名的其他MindRecord文件将会被自动寻找并加载。如果 `dataset_files` 的类型是列表，则它表示所需读取的MindRecord数据文件。
        - **columns_list** (list[str]，可选) - 指定从MindRecord文件中读取的数据列。默认值：None，读取所有列。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用全局默认线程数(8)，也可以通过 `mindspore.dataset.config.set_num_parallel_workers` 配置全局线程数。
        - **shuffle** (Union[bool, Shuffle], 可选) - 每个epoch中数据混洗的模式，支持传入bool类型与枚举类型进行指定。默认值：mindspore.dataset.Shuffle.GLOBAL。
          如果 `shuffle` 为False，则不混洗，如果 `shuffle` 为True，等同于将 `shuffle` 设置为mindspore.dataset.Shuffle.GLOBAL。
          通过传入枚举变量设置数据混洗的模式：

          - **Shuffle.GLOBAL**：混洗文件和文件中的数据。
          - **Shuffle.FILES**：仅混洗文件。
          - **Shuffle.INFILE**：保持读入文件的序列，仅混洗每个文件中的数据。

        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值：None。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值：None。只有当指定了 `num_shards` 时才能指定此参数。
        - **sampler** (Sampler, 可选) - 指定从数据集中选取样本的采样器。默认值：None。下表中会展示不同配置的预期行为。当前此数据集仅支持以下采样器：SubsetRandomSampler、PkSampler、RandomSampler、SequentialSampler和DistributedSampler。
        - **padded_sample** (dict, 可选) - 指定额外添加到数据集的样本，可用于在分布式训练时补齐分片数据，注意字典的键名需要与 `column_list` 指定的列名相同。默认值：None，不添加样本。需要与 `num_padded` 参数同时使用。
        - **num_padded** (int, 可选) - 指定额外添加的数据集样本的数量。在分布式训练时可用于为数据集补齐样本，使得总样本数量可被 `num_shards` 整除。默认值：None，不添加样本。需要与 `padded_sample` 参数同时使用。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None，读取所有样本。
        - **cache** (DatasetCache, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/cache.html>`_ 。默认值：None，不使用缓存。

    异常：
        - **ValueError** - `dataset_files` 参数所指向的文件无效或不存在。
        - **ValueError** - `num_parallel_workers` 参数超过最大线程数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - 如果 `shard_id` 取值不在[0, `num_shards` )范围。

    .. note:: 此数据集可以指定参数 `sampler` ，但参数 `sampler` 和参数 `shuffle` 的行为是互斥的。下表展示了几种合法的输入参数组合及预期的行为。

    .. list-table:: 配置 `sampler` 和 `shuffle` 的不同组合得到的预期排序结果
       :widths: 25 25 50
       :header-rows: 1

       * - 参数 `sampler`
         - 参数 `shuffle`
         - 预期数据顺序
       * - None
         - None
         - 随机排列
       * - None
         - True
         - 随机排列
       * - None
         - False
         - 顺序排列
       * - `sampler` 实例
         - None
         - 由 `sampler` 行为定义的顺序
       * - `sampler` 实例
         - True
         - 不允许
       * - `sampler` 实例
         - False
         - 不允许


.. include:: mindspore.dataset.api_list_nlp.rst
