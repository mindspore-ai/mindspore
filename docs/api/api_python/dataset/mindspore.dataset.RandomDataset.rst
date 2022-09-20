mindspore.dataset.RandomDataset
===============================

.. py:class:: mindspore.dataset.RandomDataset(total_rows=None, schema=None, columns_list=None, num_samples=None, num_parallel_workers=None, cache=None, shuffle=None, num_shards=None, shard_id=None)

    生成随机数据的源数据集。

    参数：
        - **total_rows** (int, 可选) - 随机生成样本数据的数量。默认值：None，生成随机数量的样本。
        - **schema** (Union[str, Schema], 可选) - 读取模式策略，用于指定读取数据列的数据类型、数据维度等信息。
          支持传入JSON文件路径或 mindspore.dataset.Schema 构造的对象。默认值：None，不指定。
        - **columns_list** (list[str], 可选) - 指定生成数据集的列名，默认值：None，生成的数据列将以"c0"，"c1"，"c2" ... "cn"的规则命名。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None，读取所有样本。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用mindspore.dataset.config中配置的线程数。
        - **cache** (DatasetCache, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/cache.html>`_ 。默认值：None，不使用缓存。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值：None，下表中会展示不同参数配置的预期行为。
        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数，默认值：None。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号，默认值：None。只有当指定了 `num_shards` 时才能指定此参数。


.. include:: mindspore.dataset.api_list_nlp.rst
