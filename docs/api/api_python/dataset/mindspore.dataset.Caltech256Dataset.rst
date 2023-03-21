mindspore.dataset.Caltech256Dataset
===================================

.. py:class:: mindspore.dataset.Caltech256Dataset(dataset_dir, num_samples=None, num_parallel_workers=None, shuffle=None, decode=False, sampler=None, num_shards=None, shard_id=None, cache=None)

    Caltech 256数据集。

    生成的数据集有两列 `[image, label]` 。 `image` 列的数据类型为uint8。`label` 列的数据类型为uint32。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数，可以小于数据集总数。默认值：None，读取全部样本图片。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用全局默认线程数(8)，也可以通过 `mindspore.dataset.config.set_num_parallel_workers` 配置全局线程数。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值：None。下表中会展示不同参数配置的预期行为。
        - **decode** (bool, 可选) - 是否对读取的图片进行解码操作。默认值：False，不解码。
        - **sampler** (Sampler, 可选) - 指定从数据集中选取样本的采样器。默认值：None。下表中会展示不同配置的预期行为。
        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值：None。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值：None。只有当指定了 `num_shards` 时才能指定此参数。
        - **cache** (DatasetCache, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/cache.html>`_ 。默认值：None，不使用缓存。

    异常：
        - **RuntimeError** - `dataset_dir` 路径下不包含任何数据文件。
        - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 参数。
        - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 参数或同时指定了 `sampler` 和 `shard_id` 参数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - 如果 `shard_id` 取值不在[0, `num_shards` )范围。
        - **ValueError** - `target_type` 参数取值不为 'category'、 'annotation'或 'all'。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。

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

    **关于Caltech256数据集：**

    Caltech-256 是一个对象识别数据集，包含 30,607 张不同大小的真实世界图像，共有 257 个类别（256类物体和1个其他类），
    每个类别由至少 80 张图像。该数据集是 Caltech101 数据集的超集。

    您可以解压缩原始Caltech256数据集文件到如下目录结构，并通过MindSpore的API进行读取。

    .. code-block::

        .
        └── caltech256_dataset_directory
             ├── 001.ak47
             │    ├── 001_0001.jpg
             │    ├── 001_0002.jpg
             │    ...
             ├── 002.american-flag
             │    ├── 002_0001.jpg
             │    ├── 002_0002.jpg
             │    ...
             ├── 003.backpack
             │    ├── 003_0001.jpg
             │    ├── 003_0002.jpg
             │    ...
             ├── ...

    **引用：**

    .. code-block::

        @article{griffin2007caltech,
        title     = {Caltech-256 object category dataset},
        added-at  = {2021-01-21T02:54:42.000+0100},
        author    = {Griffin, Gregory and Holub, Alex and Perona, Pietro},
        biburl    = {https://www.bibsonomy.org/bibtex/21f746f23ff0307826cca3e3be45f8de7/s364315},
        interhash = {bfe1e648c1778c04baa60f23d1223375},
        intrahash = {1f746f23ff0307826cca3e3be45f8de7},
        publisher = {California Institute of Technology},
        timestamp = {2021-01-21T02:54:42.000+0100},
        year      = {2007}
        }


.. include:: mindspore.dataset.api_list_vision.rst
