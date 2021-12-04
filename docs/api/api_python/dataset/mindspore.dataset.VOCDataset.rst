mindspore.dataset.VOCDataset
=============================

.. py:class:: mindspore.dataset.VOCDataset(dataset_dir, task='Segmentation', usage='train', class_indexing=None, num_samples=None, num_parallel_workers=None, shuffle=None, decode=False, sampler=None, num_shards=None, shard_id=None, cache=None, extra_metadata=False)

    用于读取和解析VOC数据集的源数据集文件。

    根据给定的task配置，数据集会生成不同的输出列：

    - task = `Detection`，输出列： `[image, dtype=uint8]`, `[bbox, dtype=float32]`, `[label, dtype=uint32]`, `[difficult, dtype=uint32]`, `[truncate, dtype=uint32]`。
    - task = `Segmentation`，输出列： `[image, dtype=uint8]`, `[target,dtype=uint8]`。

    **参数：**

    - **dataset_dir** (str): 包含数据集文件的根目录的路径。
    - **task** (str, 可选): 指定读取VOC数据的任务类型，现在只支持 `Segmentation` 或 `Detection` （默认值 `Segmentation` ）。
    - **usage** (str, 可选): 指定数据集的子集（默认值 `train` ）。如果 `task` 参数为 `Segmentation` ，则将在./ImageSets/Segmentation/usage + ".txt"中加载数据集图像和标注信息；如果 `task` 参数为 `Detection` ，则将在./ImageSets/Main/usage + ".txt"中加载数据集图像和标注信息；如果未设置任务和用法，默认将加载./ImageSets/Segmentation/train.txt中的数据集图像和标注信息。
    - **class_indexing** (dict, 可选): 指定标签名称到类标签的映射，要求映射规则为str到int，仅在 `Detection` 任务中有效（默认值None，文件夹名称将按字母顺序排列，每类都有一个唯一的索引，从0开始)。
    - **num_samples** (int, 可选): 指定从数据集中读取的样本数（默认值为None，所有图像样本）。
    - **num_parallel_workers** (int, 可选): 指定读取数据的工作线程数（默认值None，即使用mindspore.dataset.config中配置的线程数）。
    - **shuffle** (bool, 可选): 是否混洗数据集（默认为None，下表中会展示不同配置的预期行为）。
    - **decode** (bool, 可选): 是否对读取的图像进行解码操作（默认值为False）。
    - **sampler** (Sampler, 可选): 指定从数据集中选取样本的采样器（默认为None，下表中会展示不同配置的预期行为）。
    - **num_shards** (int, 可选): 分布式训练时，将数据集划分成指定的分片数（默认值None）。指定此参数后， `num_samples` 表示每个分片的最大样本数。
    - **shard_id** (int, 可选): 分布式训练时，指定使用的分片ID号（默认值None）。只有当指定了 `num_shards` 时才能指定此参数。
    - **cache** (DatasetCache, 可选): 数据缓存客户端实例，用于加快数据集处理速度（默认为None，不使用缓存）。
    - **extra_metadata** (bool, 可选): 用于指定是否额外输出一列数据用于表示图像元信息。如果为True，则将额外输出一列数据，名为 `[_meta-filename, dtype=string]` （默认值为False）。

    **异常：**

    - **RuntimeError** - `dataset_dir` 不包含任何数据文件。
    - **RuntimeError** - `num_parallel_workers` 超过系统最大线程数。
    - **RuntimeError** - 标注的xml文件格式异常或无效。
    - **RuntimeError** - 标注的xml文件缺失 `object` 属性。
    - **RuntimeError** - 标注的xml文件缺失 `bndbox` 属性。
    - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 。
    - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 。
    - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
    - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
    - **ValueError** - 指定的任务不为'Segmentation'或'Detection'。
    - **ValueError** - 指定任务为'Segmentation'时，class_indexing不为None。
    - **ValueError** - 与usage相关的txt文件不存在。
    - **ValueError** -  `shard_id` 参数错误（小于0或者大于等于 `num_shards` ）。

    .. note::
        - 当指定 `extra_metadata` 为True时，除非显式使用rename算子以删除元信息列明的前缀('_meta-')，否则迭代的数据行中不会出现'[_meta-filename, dtype=string]'列。
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

    >>> voc_dataset_dir = "/path/to/voc_dataset_directory"
    >>>
    >>> # 1) 读取VOC数据的Segmentation任务中的train部分进行训练
    >>> dataset = ds.VOCDataset(dataset_dir=voc_dataset_dir, task="Segmentation", usage="train")
    >>>
    >>> # 2) 读取VOC数据的Detection任务中的train部分进行训练
    >>> dataset = ds.VOCDataset(dataset_dir=voc_dataset_dir, task="Detection", usage="train")
    >>>
    >>> # 3) 以8个线程随机顺序读取voc_dataset_dir中的所有VOC数据集样本
    >>> dataset = ds.VOCDataset(dataset_dir=voc_dataset_dir, task="Detection", usage="train",
    ...                         num_parallel_workers=8)
    >>>
    >>> # 4) 读voc_dataset_dir中的所有VOC数据集图片样本，且对图像进行解码
    >>> dataset = ds.VOCDataset(dataset_dir=voc_dataset_dir, task="Detection", usage="train",
    ...                         decode=True, shuffle=False)
    >>>
    >>> # 在VOC数据集中，如果task='Segmentation'，每一次迭代得到的数据行都有"image"和"target"两个键。
    >>> # 在VOC数据集中，如果task='Detection'，每一次迭代得到的数据行都有"image"和"annotation"两个键。

    **关于VOC数据集：**

    PASCAL Visual Object Classes（VOC）是视觉目标识别和检测的挑战赛，它为视觉和机器学习社区提供了图像和标注的标准数据集，称为VOC数据集。

    您可以解压缩原始VOC-2012数据集文件到如下目录结构，并通过MindSpore的API进行读取。

    .. code-block::

        .
        └── voc2012_dataset_dir
            ├── Annotations
            │    ├── 2007_000027.xml
            │    ├── 2007_000032.xml
            │    ├── ...
            ├── ImageSets
            │    ├── Action
            │    ├── Layout
            │    ├── Main
            │    └── Segmentation
            ├── JPEGImages
            │    ├── 2007_000027.jpg
            │    ├── 2007_000032.jpg
            │    ├── ...
            ├── SegmentationClass
            │    ├── 2007_000032.png
            │    ├── 2007_000033.png
            │    ├── ...
            └── SegmentationObject
                 ├── 2007_000032.png
                 ├── 2007_000033.png
                 ├── ...

    **引用：**

    .. code-block::

        @article{Everingham10,
        author       = {Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.},
        title        = {The Pascal Visual Object Classes (VOC) Challenge},
        journal      = {International Journal of Computer Vision},
        volume       = {88},
        year         = {2012},
        number       = {2},
        month        = {jun},
        pages        = {303--338},
        biburl       = {http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.html#bibtex},
        howpublished = {http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html}
        }

    .. include:: mindspore.dataset.Dataset.add_sampler.rst

    .. include:: mindspore.dataset.Dataset.rst

    .. include:: mindspore.dataset.Dataset.use_sampler.rst

    .. include:: mindspore.dataset.Dataset.zip.rst