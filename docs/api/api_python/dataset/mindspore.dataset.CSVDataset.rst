mindspore.dataset.CSVDataset
=============================

.. py:class:: mindspore.dataset.CSVDataset(dataset_files, field_delim=',', column_defaults=None, column_names=None, num_samples=None, num_parallel_workers=None, shuffle=<Shuffle.GLOBAL: 'global'>, num_shards=None, shard_id=None, cache=None)

    读取和解析CSV数据文件构建数据集。生成的数据集的列名和列类型取决于输入的CSV文件。

    **参数：**

    - **dataset_files** (Union[str, list[str]]) - 数据集文件路径，支持单文件路径字符串、多文件路径字符串列表或可被glob库模式匹配的字符串，文件列表将在内部进行字典排序。
    - **field_delim** (str, 可选) - 指定用于分隔字段的分隔符，默认值：','。
    - **column_defaults** (list, 可选) - 指定每个数据列的数据类型，有效的类型包括float、int或string。默认值：None，不指定。如果未指定该参数，则所有列的数据类型将被视为string。
    - **column_names** (list[str], 可选) - 指定数据集生成的列名。默认值：None，不指定。如果未指定该列表，则将CSV文件首行提供的字段作为列名生成。
    - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None，读取全部样本。
    - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用mindspore.dataset.config中配置的线程数。
    - **shuffle** (Union[bool, Shuffle], 可选) - 每个epoch中数据混洗的模式，支持传入bool类型与枚举类型进行指定，默认值：mindspore.dataset.Shuffle.GLOBAL。
      如果`shuffle`为False，则不混洗，如果`shuffle`为True，等同于将 `shuffle` 设置为mindspore.dataset.Shuffle.GLOBAL。
      通过传入枚举变量设置数据混洗的模式：

      - **Shuffle.GLOBAL**：混洗文件和文件中的数据。
      - **Shuffle.FILES**：仅混洗文件。

    - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数，默认值：None。指定此参数后, `num_samples` 表示每个分片的最大样本数。
    - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号，默认值：None。只有当指定了 `num_shards` 时才能指定此参数。
    - **cache** (DatasetCache, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/docs/programming_guide/zh-CN/master/cache.html>`_ 。默认值：None，不使用缓存。

    **异常：**

    - **RuntimeError** - `dataset_files` 参数所指向的文件无效或不存在。
    - **RuntimeError** - `num_parallel_workers` 参数超过系统最大线程数。
    - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
    - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。

    **样例：**

    >>> csv_dataset_dir = ["/path/to/csv_dataset_file"] # 此列表可以包含一个或多个CSV文件
    >>> dataset = ds.CSVDataset(dataset_files=csv_dataset_dir, column_names=['col1', 'col2', 'col3', 'col4'])

    .. include:: mindspore.dataset.Dataset.rst

    .. include:: mindspore.dataset.Dataset.zip.rst
