mindspore.dataset.ImageFolderDataset
=====================================

.. py:class:: mindspore.dataset.ImageFolderDataset(dataset_dir, num_samples=None, num_parallel_workers=None, shuffle=None, sampler=None, extensions=None, class_indexing=None, decode=False, num_shards=None, shard_id=None, cache=None)

    从树状结构的文件目录中读取图像作为源数据集，同一个文件夹中的所有图像都具有相同的标签。

    生成的数据集有两列：`[image, label]`。列: `image` 的数据为uint8类型，列: `label` 的数据是uint32类型的标量。

    **参数：**

    - **dataset_dir** (str) - 包含数据集文件的根目录的路径。
    - **num_samples** (int, 可选) - 指定从数据集中读取的样本数（可以小于数据集总数，默认值为None，即全部样本图片）。
    - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数（默认值None，即使用 `mindspore.dataset.config` 中配置的线程数）。
    - **shuffle** (bool, 可选) - 是否混洗数据集（默认为None，下表中会展示不同配置的预期行为）。
    - **sampler** (Sampler, 可选) - 指定从数据集中选取样本的采样器（默认为None，下表中会展示不同配置的预期行为）。
    - **extensions** (list[str], 可选) - 指定文件扩展后缀，仅读取这些后续的文件到数据集中（默认为None）。
    - **class_indexing** (dict, 可选) - 指定文件夹名称到类标签的映射，要求映射规则为str到int（默认为None，文件夹名称将按字母顺序排列，每类都有一个唯一的索引，从0开始）。
    - **decode** (bool, 可选) - 是否对读取的图像进行解码操作（默认为False）。
    - **num_shards** (int, 可选) - 分布式训练时，将数据集划分成指定的分片数（默认值None）。指定此参数后，`num_samples` 表示每个分片的最大样本数。
    - **shard_id** (int, 可选) - 分布式训练时，指定使用的分片ID号（默认值None）。只有当指定了 `num_shards` 时才能指定此参数。
    - **cache** (DatasetCache, 可选) - 单节点数据缓存，能够加快数据加载和处理的速度（默认值None, 即不使用缓存加速）。

    **异常：**

    - **RuntimeError** - `dataset_dir` 不包含任何数据文件。
    - **RuntimeError** - `num_parallel_workers` 超过系统最大线程数。
    - **RuntimeError** - 同时指定了采样器和 `shuffle` 。
    - **RuntimeError** - 同时指定了采样器和分片。
    - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
    - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
    - **RuntimeError** - `class_indexing` 的类型不是字典。
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

    >>> image_folder_dataset_dir = "/path/to/image_folder_dataset_directory"
    >>>
    >>> # 1）使用8个线程读取image_folder_dataset_dir中的所有图像文件。
    >>> dataset = ds.ImageFolderDataset(dataset_dir=image_folder_dataset_dir,
    ...                                 num_parallel_workers=8)
    >>>
    >>> # 2）从标签为0和1的cat文件夹为和dog文件夹中读取所有图像文件。
    >>> dataset = ds.ImageFolderDataset(dataset_dir=image_folder_dataset_dir,
    ...                                 class_indexing={"cat":0, "dog":1})
    >>>
    >>> # 3）读取image_folder_dataset_dir中所有扩展名为.JPEG和.png（区分大小写）的图像文件。
    >>> dataset = ds.ImageFolderDataset(dataset_dir=image_folder_dataset_dir,
    ...                                 extensions=[".JPEG", ".png"])

    **关于ImageFolderDataset：**

    您可以将图像数据文件构建成如下目录结构，并通过MindSpore的API进行读取。

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

    .. include:: mindspore.dataset.Dataset.add_sampler.rst

    .. include:: mindspore.dataset.Dataset.rst

    .. include:: mindspore.dataset.Dataset.use_sampler.rst

    .. include:: mindspore.dataset.Dataset.zip.rst
