mindspore.dataset.ImageFolderDataset
=====================================

.. py:class:: mindspore.dataset.ImageFolderDataset(dataset_dir, num_samples=None, num_parallel_workers=None, shuffle=None, sampler=None, extensions=None, class_indexing=None, decode=False, num_shards=None, shard_id=None, cache=None, decrypt=None)

    从树状结构的文件目录中读取图片构建源数据集。同一个文件夹中的所有图片将被分配相同的label。

    生成的数据集有两列：`[image, label]`。`image` 列的数据类型为uint8。 `label` 列的数据类型为uint32。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录的路径。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数，可以小于数据集总数。默认值：None，读取全部样本图片。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用 `mindspore.dataset.config` 中配置的线程数。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值：None。下表中会展示不同参数配置的预期行为。
        - **sampler** (Sampler, 可选) - 指定从数据集中选取样本的采样器。默认值：None。下表中会展示不同配置的预期行为。
        - **extensions** (list[str], 可选) - 指定文件的扩展名，仅读取与指定扩展名匹配的文件到数据集中。默认值：None。
        - **class_indexing** (dict, 可选) - 指定文件夹名称到label索引的映射，要求映射规则为string到int。文件夹名称将按字母顺序排列，索引值从0开始，并且要求每个文件夹名称对应的索引值唯一。默认值：None，不指定。
        - **decode** (bool, 可选) - 是否对读取的图片进行解码操作。默认值：False，不解码。
        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值：None。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值：None。只有当指定了 `num_shards` 时才能指定此参数。
        - **cache** (DatasetCache, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/cache.html>`_ 。默认值：None，不使用缓存。
        - **decrypt** (callable, 可选) - 图像解密函数，接受加密的图片路径并返回bytes类型的解密数据。默认值：None，不进行解密。

    异常：
        - **RuntimeError** - `dataset_dir` 不包含任何数据文件。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。
        - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 参数。
        - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 参数或同时指定了 `sampler` 和 `shard_id` 参数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **RuntimeError** - `class_indexing` 参数的类型不是dict。
        - **ValueError** - 如果 `shard_id` 取值不在[0, `num_shards` )范围。

    .. note::
        - 如果 `decode` 参数的值为False，则得到的 `image` 列的shape为[undecoded_image_size]，如果为True则 `image` 列的shape为[H,W,C]。
        - 此数据集可以指定参数 `sampler` ，但参数 `sampler` 和参数 `shuffle` 的行为是互斥的。下表展示了几种合法的输入参数组合及预期的行为。

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

    **关于ImageFolderDataset：**

    您可以将图片数据文件构建成如下目录结构，并通过MindSpore的API进行读取。

    .. code-block::

        .
        └── image_folder_dataset_directory
             ├── class1
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
             ├── class2
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
             ├── class3
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
             ├── classN
             ├── ...


.. include:: mindspore.dataset.api_list_vision.rst
