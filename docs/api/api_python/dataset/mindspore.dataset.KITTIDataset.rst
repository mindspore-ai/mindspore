mindspore.dataset.KITTIDataset
==============================

.. py:class:: mindspore.dataset.KITTIDataset(dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None, decode=False, sampler=None, num_shards=None, shard_id=None, cache=None)

    KITTI数据集。

    当 `usage` 为 'train' 时，生成的数据集有多列: `[image, label, truncated, occluded, alpha, bbox, dimensions, location, rotation_y]` ；当 `usage` 为 'test' 时，生成的数据集只有一列 `[image]` 。
    `image` 列的数据类型为uint8。
    `label` 列的数据类型为uint32。
    `truncated` 列的数据类型为float32。
    `occluded` 列的数据类型为uint32。
    `alpha` 列的数据类型为float32。
    `bbox` 列的数据类型为float32。
    `dimensions` 列的数据类型为float32。
    `location` 列的数据类型为float32。
    `rotation_y` 列的数据类型为float32。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 'train' 或 'test'。取值为 'train' 时将会读取7481个训练样本，取值为 'test' 时将会读取7518个不带标签的测试样本。默认值：None，将使用 'train'。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数，可以小于数据集总数。默认值：None，读取全部样本图片。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用全局默认线程数(8)，也可以通过 `mindspore.dataset.config.set_num_parallel_workers` 配置全局线程数。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值：None。下表中会展示不同参数配置的预期行为。
        - **decode** (bool, 可选) - 是否对读取的图片进行解码操作。默认值：False，不解码。
        - **sampler** (Sampler, 可选) - 指定从数据集中选取样本的采样器。默认值：None。下表中会展示不同配置的预期行为。
        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值：None。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值：None。只有当指定了 `num_shards` 时才能指定此参数。
        - **cache** (DatasetCache, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/cache.html>`_ 。默认值：None，不使用缓存。

    异常：
        - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 参数。
        - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 参数或同时指定了 `sampler` 和 `shard_id` 参数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - `dataset_dir` 对应目录不存在。
        - **ValueError** - 如果 `shard_id` 取值不在[0, `num_shards` )范围。

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

    **关于KITTI数据集：**
    
    KITTI（Karlsruhe Institute of Technology and Toyota Technological Institute）是移动机器人和自动驾驶领域的常用数据集之一。
    它由用高分辨率的RGB、灰度立体相机和三维激光扫描仪等各种传感器记录的数小时的交通场景组成。
    尽管它很常用，但该数据集本身并不包含用于语义分割的目标值。然而，许多研究人员已经对该数据集的部分内容进行了手工标注以适应需求。
    Álvarez等人为道路检测挑战中的323幅图像生成了目标值，包含3个类别：道路、车辆和天空。
    Zhang等人为追踪挑战中的252张（140张用于训练，112张用于测试）RGB和激光扫描图像进行了标注，包含10个类别：建筑、天空、道路、植被、人行道、汽车、行人、自行车、标志/杆、栅栏。

    您可以解压原始KITTI数据集文件构建成如下目录结构，并通过MindSpore的API进行读取。

    .. code-block::

        .
        └── kitti_dataset_directory
            ├── data_object_image_2
            │    ├──training
            │    │    ├──image_2
            │    │    │    ├── 000000000001.jpg
            │    │    │    ├── 000000000002.jpg
            │    │    │    ├── ...
            │    ├──testing
            │    │    ├── image_2
            │    │    │    ├── 000000000001.jpg
            │    │    │    ├── 000000000002.jpg
            │    │    │    ├── ...
            ├── data_object_label_2
            │    ├──training
            │    │    ├──label_2
            │    │    │    ├── 000000000001.jpg
            │    │    │    ├── 000000000002.jpg
            │    │    │    ├── ...

    **引用：**

    .. code-block::

        @INPROCEEDINGS{Geiger2012CVPR,
        author={Andreas Geiger and Philip Lenz and Raquel Urtasun},
        title={Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
        booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2012}
        }


.. include:: mindspore.dataset.api_list_vision.rst
