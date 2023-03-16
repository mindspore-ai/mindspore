mindspore.dataset.Caltech101Dataset
===================================

.. py:class:: mindspore.dataset.Caltech101Dataset(dataset_dir, target_type=None, num_samples=None, num_parallel_workers=1, shuffle=None, decode=False, sampler=None, num_shards=None, shard_id=None)

    读取和解析Caltech101数据集的源文件构建数据集。

    根据不同的 `target_type` 配置，数据集会生成不同的输出列。

    - `target_type` = 'category'，输出列为 `[image, category]` 。 
    - `target_type` = 'annotation'，输出列为 `[image, annotation]` 。
    - `target_type` = 'all'，输出列为 `[image, category, annotation]` 。
  
    列 'image' 为 uint8 类型。列 'category' 为 uint32 类型。列 'annotation' 是一个二维的ndarray，存储了图像的轮廓，由一系列的点组成。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径，该路径下将包含2个子目录，目录101_ObjectCategories用于存储图像，
          目录Annotations用于存储图像的标注。
        - **target_type** (str, 可选) - 指定数据集的子集，可取值为 'category'、'annotation' 或 'all'。
          取值为 'category'时将读取图像的类别标注作为label，取值为 'annotation'时将读取图像的轮廓标注作为label，
          取值为 'all'时将同时输出图像的类别标注和轮廓标注。默认值：None，表示 'category'。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数，可以小于数据集总数。默认值：None，读取全部样本图片。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用 `mindspore.dataset.config` 中配置的线程数。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值：None。下表中会展示不同参数配置的预期行为。
        - **decode** (bool, 可选) - 是否对读取的图片进行解码操作。默认值：False，不解码。
        - **sampler** (Sampler, 可选) - 指定从数据集中选取样本的采样器。默认值：None。下表中会展示不同配置的预期行为。
        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值：None。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值：None。只有当指定了 `num_shards` 时才能指定此参数。

    异常：
        - **RuntimeError** - `dataset_dir` 路径下不包含任何数据文件。
        - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 参数。
        - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 参数或同时指定了 `sampler` 和 `shard_id` 参数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - `shard_id` 参数错误，小于0或者大于等于 `num_shards` 。
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

    **关于Caltech101数据集：**

    Caltech101数据集包含 101 种类别的图片。每种类别大约 40 到 800 张图像，大多数类别有大约 50 张图像。 
    每张图像的大小约为 300 x 200 像素。数据集中也提供了每张图片中每个物体的轮廓数据，用于检测和定位。

    您可以解压缩原始Caltech101数据集文件到如下目录结构，并通过MindSpore的API进行读取。

    .. code-block::

        .
        └── caltech101_dataset_directory
            ├── 101_ObjectCategories
            │    ├── Faces
            │    │    ├── image_0001.jpg
            │    │    ├── image_0002.jpg
            │    │    ...
            │    ├── Faces_easy
            │    │    ├── image_0001.jpg
            │    │    ├── image_0002.jpg
            │    │    ...
            │    ├── ...
            └── Annotations
                 ├── Airplanes_Side_2
                 │    ├── annotation_0001.mat
                 │    ├── annotation_0002.mat
                 │    ...
                 ├── Faces_2
                 │    ├── annotation_0001.mat
                 │    ├── annotation_0002.mat
                 │    ...
                 ├── ...

    **引用：**

    .. code-block::

        @article{FeiFei2004LearningGV,
        author    = {Li Fei-Fei and Rob Fergus and Pietro Perona},
        title     = {Learning Generative Visual Models from Few Training Examples:
                    An Incremental Bayesian Approach Tested on 101 Object Categories},
        journal   = {Computer Vision and Pattern Recognition Workshop},
        year      = {2004},
        url       = {http://data.caltech.edu/records/20086},
        }


.. include:: mindspore.dataset.api_list_vision.rst
