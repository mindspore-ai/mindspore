mindspore.dataset.Caltech256Dataset
===================================

.. py:class:: mindspore.dataset.Caltech256Dataset(dataset_dir, num_samples=None, num_parallel_workers=None, shuffle=None, decode=False, sampler=None, num_shards=None, shard_id=None, cache=None)

    Caltech 256数据集。

    生成的数据集有两列 `[image, label]` 。 `image` 列的数据类型为uint8。 `label` 列的数据类型为uint32。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数，可以小于数据集总数。默认值： ``None`` ，读取全部样本图片。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值： ``None`` ，使用全局默认线程数（8），也可以通过 :func:`mindspore.dataset.config.set_num_parallel_workers` 配置全局线程数。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值： ``None`` 。下表中会展示不同参数配置的预期行为。
        - **decode** (bool, 可选) - 是否对读取的图片进行解码操作。默认值： ``False`` ，不解码。
        - **sampler** (Sampler, 可选) - 指定从数据集中选取样本的采样器。默认值： ``None`` 。下表中会展示不同配置的预期行为。
        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值： ``None`` 。指定此参数后， `num_samples` 表示每个分片的最大样本数。一般在 `数据并行模式训练 <https://www.mindspore.cn/tutorials/zh-CN/master/parallel/data_parallel.html#数据集加载>`_ 的时候使用。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值： ``None`` 。只有当指定了 `num_shards` 时才能指定此参数。
        - **cache** (:class:`~.dataset.DatasetCache`, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/zh-CN/master/dataset/cache.html>`_ 。默认值： ``None`` ，不使用缓存。

    异常：
        - **RuntimeError** - `dataset_dir` 路径下不包含任何数据文件。
        - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 参数。
        - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 参数，或同时指定了 `sampler` 和 `shard_id` 参数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - 如果 `shard_id` 取值不在[0, `num_shards` )范围。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。

    教程样例：
        - `使用数据Pipeline加载 & 处理数据集
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/dataset_gallery.html>`_

    .. note:: 入参 `num_samples` 、 `shuffle` 、 `num_shards` 、 `shard_id` 可用于控制数据集所使用的采样器，其与入参 `sampler` 搭配使用的效果如下。

    .. include:: mindspore.dataset.sampler.rst

    **关于Caltech256数据集：**

    Caltech-256 是一个对象识别数据集，包含 30,607 张不同大小的真实世界图像，共有 257 个类别（256类物体和1个其他类），
    每个类别至少有 80 张图像。该数据集是 Caltech101 数据集的超集。

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
