mindspore.dataset.VOCDataset
=============================

.. py:class:: mindspore.dataset.VOCDataset(dataset_dir, task='Segmentation', usage='train', class_indexing=None, num_samples=None, num_parallel_workers=None, shuffle=None, decode=False, sampler=None, num_shards=None, shard_id=None, cache=None, extra_metadata=False, decrypt=None)

    读取和解析VOC数据集的源文件构建数据集。

    根据给定的 `task` 配置，生成数据集具有不同的输出列：

    - `task` = 'Detection'，输出列： `[image, dtype=uint8]` , `[bbox, dtype=float32]` , `[label, dtype=uint32]` , `[difficult, dtype=uint32]` , `[truncate, dtype=uint32]` 。
    - `task` = 'Segmentation'，输出列： `[image, dtype=uint8]` , `[target, dtype=uint8]` 。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录的路径。
        - **task** (str, 可选) - 指定读取VOC数据的任务类型，现在只支持 'Segmentation' 和 'Detection'。默认值：'Segmentation'。
        - **usage** (str, 可选) - 指定数据集的子集。默认值：'train'。

          - 如果 'task' 的值为 'Segmentation'，则读取 'ImageSets/Segmentation/' 目录下定义的图片和label信息；
          - 如果 'task' 的值为 'Detection' ，则读取 'ImageSets/Main/' 目录下定义的图片和label信息。
        - **class_indexing** (dict, 可选) - 指定一个从label名称到label索引的映射，要求映射规则为string到int。索引值从0开始，并且要求每个label名称对应的索引值唯一。
          仅在 'Detection' 任务中有效。默认值：None，不指定。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None，所有图像样本。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用mindspore.dataset.config中配置的线程数。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值：None。下表中会展示不同参数配置的预期行为。
        - **decode** (bool, 可选) - 是否对读取的图片进行解码操作。默认值：False，不解码。
        - **sampler** (Sampler, 可选) - 指定从数据集中选取样本的采样器。默认值：None。下表中会展示不同配置的预期行为。
        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值：None。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值：None。只有当指定了 `num_shards` 时才能指定此参数。
        - **cache** (DatasetCache, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/cache.html>`_ 。默认值：None，不使用缓存。
        - **extra_metadata** (bool, 可选) - 用于指定是否额外输出一个数据列用于表示图片元信息。如果为True，则将额外输出一个名为 `[_meta-filename, dtype=string]` 的数据列。默认值：False。
        - **decrypt** (callable, 可选) - 图像解密函数，接受加密的图片路径并返回bytes类型的解密数据。默认值：None，不进行解密。

    异常：
        - **RuntimeError** - `dataset_dir` 路径下不包含任何数据文件。
        - **RuntimeError** - 读取的xml文件格式异常或无效。
        - **RuntimeError** - 读取的xml文件缺失 `object` 属性。
        - **RuntimeError** - 读取的xml文件缺失 `bndbox` 属性。
        - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 参数。
        - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 参数或同时指定了 `sampler` 和 `shard_id` 参数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。
        - **ValueError** - 指定的任务不为 'Segmentation'或 'Detection'。
        - **ValueError** - 指定任务为 'Segmentation'时， `class_indexing` 参数不为None。
        - **ValueError** - 与 `usage` 参数相关的txt文件不存在。
        - **ValueError** - `shard_id` 参数错误，小于0或者大于等于 `num_shards` 。

    .. note::
        - 当参数 `extra_metadata` 为True时，还需使用 `rename` 操作删除额外数据列 '_meta-filename'的前缀 '_meta-'，
          否则迭代得到的数据行中不会出现此额外数据列。
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


.. include:: mindspore.dataset.api_list_vision.rst
