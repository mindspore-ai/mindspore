mindspore.dataset.TFRecordDataset
=================================

.. py:class:: mindspore.dataset.TFRecordDataset(dataset_files, schema=None, columns_list=None, num_samples=None, num_parallel_workers=None, shuffle=<Shuffle.GLOBAL: 'global'>, num_shards=None, shard_id=None, shard_equal_rows=False, cache=None)

    读取和解析以TFData格式存储的数据集文件作为源数据集。生成的数据集的列名和列类型取决于TFRecord文件中的保存的列名与类型。

    **参数：**

    - **dataset_files** (Union[str, list[str]]) - 数据集文件路径，支持单文件路径字符串、多文件路径字符串列表或可被glob库模式匹配的字符串，文件列表将在内部进行字典排序。
    - **schema** (Union[str, Schema]，可选) - 读取模式策略，用于指定读取数据列的数据类型、数据维度等信息。支持传入JSON文件或 `Schema` 对象的路径（默认为None，将使用TFData文件中的元数据构造 `Schema` 对象）。
    - **columns_list** (list[str]，可选) - 指定从TFRecord文件中读取的数据列（默认为None，读取所有列）。
    - **num_samples** (int，可选) - 指定从数据集中读取的样本数（默认为None）。如果 `num_samples` 为None，并且numRows字段（由参数 `schema` 定义）不存在，则读取所有数据集；如果 `num_samples` 为None，并且numRows字段（由参数 `schema` 定义）的值大于0，则读取numRows条数据；如果 `num_samples` 和numRows字段（由参数 `schema` 定义）的值都大于0，仅有参数 `num_samples` 生效且读取给定数量的数据。
    - **num_parallel_workers** (int，可选) - 指定读取数据的工作线程数（默认值None，即使用mindspore.dataset.config中配置的线程数）。
    - **shuffle** (Union[bool, Shuffle level], 可选) - 每个epoch中数据混洗的模式（默认为为mindspore.dataset.Shuffle.GLOBAL）。如果为False，则不混洗；如果为True，等同于将 `shuffle` 设置为mindspore.dataset.Shuffle.GLOBAL。另外也可以传入枚举变量设置shuffle级别：

      - Shuffle.GLOBAL：混洗文件和样本。
      - Shuffle.FILES：仅混洗文件。

    - **num_shards** (int, 可选): 分布式训练时，将数据集划分成指定的分片数（默认值None）。指定此参数后，`num_samples` 表示每个分片的最大样本数。
    - **shard_id** (int, 可选): 分布式训练时，指定使用的分片ID号（默认值None）。只有当指定了 `num_shards` 时才能指定此参数。
    - **shard_equal_rows** (bool，可选)： 分布式训练时，为所有分片获取等量的数据行数（默认为False）。如果 `shard_equal_rows` 为False，则可能会使得每个分片的数据条目不相等，从而导致分布式训练失败。因此当每个TFRecord文件的数据数量不相等时，建议将此参数设置为True。注意，只有当指定了 `num_shards` 时才能指定此参数。
    - **cache** (DatasetCache, 可选) - 数据缓存客户端实例，用于加快数据集处理速度（默认为None，不使用缓存）。

    **异常：**

    - **RuntimeError** - 参数 `dataset_files` 无效或不存在。
    - **RuntimeError** - 参数 `num_parallel_workers` 超过最大线程数。
    - **RuntimeError** - 指定了 `num_shards` ，但 `shard_id` 为None。
    - **RuntimeError** - 指定了 `shard_id` ，但 `num_shards` 为None。
    - **ValueError** - 参数 `shard_id` 无效（小于0或者大于等于 `num_shards` ）。

    **样例：**

    >>> from mindspore import dtype as mstype
    >>>
    >>> tfrecord_dataset_dir = ["/path/to/tfrecord_dataset_file"] # 此列表可以包含1个或多个TFRecord文件
    >>> tfrecord_schema_file = "/path/to/tfrecord_schema_file"
    >>>
    >>> # 1) 从tfrecord_dataset_dir路径的文件读取数据集。
    >>> # 由于未指定Schema，则将TFRecord文件数据的第一行的元数据将用作Schema。
    >>> dataset = ds.TFRecordDataset(dataset_files=tfrecord_dataset_dir)
    >>>
    >>> # 2) 用户使用自定义的Schema从tfrecord_dataset_dir路径的文件读取数据集。
    >>> schema = ds.Schema()
    >>> schema.add_column(name='col_1d', de_type=mstype.int64, shape=[2])
    >>> dataset = ds.TFRecordDataset(dataset_files=tfrecord_dataset_dir, schema=schema)
    >>>
    >>> # 3) 用户通过传入JSON文件构造Schema，从tfrecord_dataset_dir路径的文件读取数据集。
    >>> dataset = ds.TFRecordDataset(dataset_files=tfrecord_dataset_dir, schema=tfrecord_schema_file)

    .. include:: mindspore.dataset.Dataset.rst

    .. include:: mindspore.dataset.Dataset.zip.rst
