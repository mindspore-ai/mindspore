mindspore.dataset.ManifestDataset
==================================

.. py:class:: mindspore.dataset.ManifestDataset(dataset_file, usage='train', num_samples=None, num_parallel_workers=None, shuffle=None, sampler=None, class_indexing=None, decode=False, num_shards=None, shard_id=None, cache=None)

    读取Manifest文件作为源数据集。

    生成的数据集有两列： `[image, label]`。列 `image` 的数据类型为uint8类型。列 `label` 的数据类型是uint64类型的标量。

    **参数：**

    - **dataset_file** (str) - 数据集文件的目录路径。
    - **usage** (str，可选) - 指定数据集的子集，可取值为'train'、'eval'和'inference' （默认为'train'）。
    - **num_samples** (int，可选) - 指定从数据集中读取的样本数（默认值为None，即全部样本图片)。
    - **num_parallel_workers** (int，可选) - 指定读取数据的工作线程数（默认值None，即使用mindspore.dataset.config中配置的线程数）。
    - **shuffle** (bool，可选) - 是否混洗数据集（默认为None，下表中会展示不同配置的预期行为）。
    - **sampler** (Sampler，可选) - 指定从数据集中选取样本的采样器（默认为None，下表中会展示不同配置的预期行为）。
    - **class_indexing** (dict，可选) - 指定文件夹名称到类标签的映射，要求映射规则为str到int（默认为None，文件夹名称将按字母顺序排列，每类都有一个唯一的索引，从0开始）。
    - **decode** (bool, 可选) - 是否对读取的图像进行解码操作（默认为False）。
    - **num_shards** (int, 可选): 分布式训练时，将数据集划分成指定的分片数（默认值None）。指定此参数后，`num_samples` 表示每个分片的最大样本数。
    - **shard_id** (int, 可选): 分布式训练时，指定使用的分片ID号（默认值None）。只有当指定了 `num_shards` 时才能指定此参数。
    - **cache** (DatasetCache, 可选) - 数据缓存客户端实例，用于加快数据集处理速度（默认为None，不使用缓存）。

    **异常：**

    - **RuntimeError** - 参数 `dataset_files` 不存在或无效。
    - **RuntimeError** - 参数 `num_parallel_workers` 超过系统最大线程数。
    - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 。
    - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 或 `shard_id`。
    - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
    - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
    - **RuntimeError** - 参数 `class_indexing` 的类型不是字典。
    - **ValueError** - `shard_id` 参数错误（小于0或者大于等于 `num_shards`）。

    .. note::
        - 如果 `decode` 参数指定为False，则 `image` 列的shape为[image_size]，否则为[H,W,C]。
        - 此数据集可以指定 `sampler` 参数，但 `sampler` 和 `shuffle` 是互斥的。下表展示了几种合法的输入参数及预期的行为。

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
       * - 参数 `sampler`
         - None
         - 由 `sampler` 行为定义的顺序
       * - 参数 `sampler`
         - True
         - 不允许
       * - 参数 `sampler`
         - False
         - 不允许

    **样例：**

    >>> manifest_dataset_dir = "/path/to/manifest_dataset_file"
    >>>
    >>> # 1）使用八个线程读取Manifest数据集文件，并指定读取"train"子集数据
    >>> dataset = ds.ManifestDataset(dataset_file=manifest_dataset_dir, usage="train", num_parallel_workers=8)
    >>>
    >>> # 2） 对Manifest数据集进行分布式训练，并将数据集拆分为2个分片，当前数据集仅加载分片ID号为0的数据
    >>> dataset = ds.ManifestDataset(dataset_file=manifest_dataset_dir, num_shards=2, shard_id=0)

    .. include:: mindspore.dataset.Dataset.add_sampler.rst

    .. include:: mindspore.dataset.Dataset.rst

    .. include:: mindspore.dataset.Dataset.use_sampler.rst

    .. include:: mindspore.dataset.Dataset.zip.rst
