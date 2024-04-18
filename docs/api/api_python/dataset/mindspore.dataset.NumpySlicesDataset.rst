mindspore.dataset.NumpySlicesDataset
=====================================

.. py:class:: mindspore.dataset.NumpySlicesDataset(data, column_names=None, num_samples=None, num_parallel_workers=1, shuffle=None, sampler=None, num_shards=None, shard_id=None)

    由Python数据构建数据集。生成的数据集的列名和列类型取决于用户传入的Python数据。

    参数：
        - **data** (Union[list, tuple, dict]) - 输入的Python数据。支持的数据类型包括：list、tuple、dict和其他NumPy格式。
          输入数据将沿着第一个维度切片，并生成额外的行。如果输入是单个list，则将生成一个数据列，若是嵌套多个list，则生成多个数据列。不建议通过这种方式加载大量的数据，因为可能会在数据加载到内存时等待较长时间。
        - **column_names** (list[str], 可选) - 指定数据集生成的列名。默认值： ``None`` ，不指定。
          如果未指定该参数，且当输入数据的类型是dict时，输出列名称将被命名为dict的键名，否则它们将被统一命名为column_0，column_1...。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值： ``None`` ，所有样本。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作进程数。默认值： ``1`` 。
        - **shuffle** (bool, 可选) - 是否混洗数据集。
          只有输入的 `data` 参数带有可随机访问属性（`__getitem__`）时，才可以指定该参数。默认值： ``None`` 。下表中会展示不同配置的预期行为。
        - **sampler** (Union[Sampler, Iterable], 可选) - 指定从数据集中选取样本的采样器。
          只有输入的 `data` 参数带有可随机访问属性（`__getitem__`）时，才可以指定该参数。默认值： ``None`` 。下表中会展示不同配置的预期行为。
        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值： ``None`` 。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值： ``None`` 。只有当指定了 `num_shards` 时才能指定此参数。

    .. note:: 入参 `num_samples` 、 `shuffle` 、 `num_shards` 、 `shard_id` 可用于控制数据集所使用的采样器，其与入参 `sampler` 搭配使用的效果如下。

    .. include:: mindspore.dataset.sampler.rst

    异常：
        - **RuntimeError** - `column_names` 列表的长度与数据的输出列表长度不匹配。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。
        - **ValueError** - 同时指定了 `sampler` 和 `shuffle` 参数。
        - **ValueError** - 同时指定了 `sampler` 和 `num_shards` 参数或同时指定了 `sampler` 和 `shard_id` 参数。
        - **ValueError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **ValueError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - 如果 `shard_id` 取值不在[0, `num_shards` )范围。

    教程样例：
        - `使用数据Pipeline加载 & 处理数据集
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/dataset_gallery.html>`_

.. include:: mindspore.dataset.api_list_nlp.rst
