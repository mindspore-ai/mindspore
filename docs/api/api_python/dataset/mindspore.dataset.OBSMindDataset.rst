mindspore.dataset.OBSMindDataset
================================

.. py:class:: mindspore.dataset.OBSMindDataset(dataset_files, server, ak, sk, sync_obs_path, columns_list=None, shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, shard_equal_rows=True)

    读取和解析存放在OBS上的MindRecord格式数据集。生成的数据集的列名和列类型取决于MindRecord文件中的保存的列名与类型。

    **参数：**

    - **dataset_files** (list[str]) - OBS上MindRecord格式数据集文件的路径列表，每个文件的路径前缀为s3://。
    - **server** (str) - 连接OBS的服务地址。可包含协议类型、域名、端口号。示例：<https://your-endpoint:9000>。
    - **ak** (str) - 访问密钥中的AK。
    - **sk** (str) - 访问密钥中的SK。
    - **sync_obs_path** (str) - 用于同步操作的OBS路径，用户需要提前创建，目录路径的前缀为s3://。
    - **columns_list** (list[str]，可选) - 指定从MindRecord文件中读取的数据列。默认值：None，读取所有列。
    - **shuffle** (Union[bool, Shuffle], 可选) - 每个epoch中数据混洗的模式，支持传入bool类型与枚举类型进行指定，默认值：mindspore.dataset.Shuffle.GLOBAL。
      如果 `shuffle` 为False，则不混洗，如果 `shuffle` 为True，等同于将 `shuffle` 设置为mindspore.dataset.Shuffle.GLOBAL。
      通过传入枚举变量设置数据混洗的模式：

      - **Shuffle.GLOBAL**：混洗文件和文件中的数据。
      - **Shuffle.FILES**：仅混洗文件。
      - **Shuffle.INFILE**：保持读入文件的序列，仅混洗每个文件中的数据。

    - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数，默认值：None。
    - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号，默认值：None。只有当指定了 `num_shards` 时才能指定此参数。
    - **shard_equal_rows** (bool, 可选) - 分布式训练时，为所有分片获取等量的数据行数。默认值：True。
      如果 `shard_equal_rows` 为False，则可能会使得每个分片的数据条目不相等，从而导致分布式训练失败。
      因此当每个TFRecord文件的数据数量不相等时，建议将此参数设置为True。注意，只有当指定了 `num_shards` 时才能指定此参数。

    **异常：**

    - **RuntimeError** - `sync_obs_path` 参数指定的目录不存在。
    - **ValueError** - `columns_list` 参数无效。
    - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
    - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
    - **ValueError** - `shard_id` 参数值错误（小于0或者大于等于 `num_shards` ）。

    .. note::
        - 需要用户提前在OBS上创建同步用的目录，然后通过 `sync_obs_path` 指定。
        - 如果线下训练，建议为每次训练设置 `BATCH_JOB_ID` 环境变量。
        - 分布式训练中，假如使用多个节点（服务器），则必须使用每个节点全部的8张卡。如果只有一个节点（服务器），则没有这样的限制。

    .. include:: mindspore.dataset.Dataset.rst

    .. include:: mindspore.dataset.Dataset.b.rst

    .. include:: mindspore.dataset.Dataset.c.rst

    .. include:: mindspore.dataset.Dataset.d.rst

    .. include:: mindspore.dataset.Dataset.zip.rst
