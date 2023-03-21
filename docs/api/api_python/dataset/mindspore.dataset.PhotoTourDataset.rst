mindspore.dataset.PhotoTourDataset
==================================

.. py:class:: mindspore.dataset.PhotoTourDataset(dataset_dir, name, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None, sampler=None, num_shards=None, shard_id=None, cache=None)

    PhotoTour数据集。

    根据给定的 `usage` 配置，生成数据集具有不同的输出列：
    - `usage` = 'train'，输出列： `[image, dtype=uint8]` 。
    - `usage` ≠ 'train'，输出列： `[image1, dtype=uint8]` 、 `[image2, dtype=uint8]` 、 `[matches, dtype=uint32]` 。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **name** (str) - 要加载的数据集内容名称，可以取值为 'notredame'、'yosemite'、'liberty'、'notredame_harris'、'yosemite_harris' 或 'liberty_harris'。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 'train' 或 'test'。默认值：None，将被设置为 'train'。
          取值为 'train'时，每个 `name` 的数据集样本数分别为{'notredame': 468159, 'yosemite': 633587, 'liberty': 450092, 'liberty_harris': 379587, 'yosemite_harris': 450912, 'notredame_harris': 325295}。
          取值为 'test'时，将读取100,000个测试样本。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None，读取所有样本。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用全局默认线程数(8)，也可以通过 `mindspore.dataset.config.set_num_parallel_workers` 配置全局线程数。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值：None。下表中会展示不同参数配置的预期行为。
        - **sampler** (Sampler, 可选) - 指定从数据集中选取样本的采样器。默认值：None。下表中会展示不同配置的预期行为。
        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值：None。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值：None。只有当指定了 `num_shards` 时才能指定此参数。
        - **cache** (DatasetCache, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/cache.html>`_ 。默认值：None，不使用缓存。

    异常：
        - **RuntimeError** - `dataset_dir` 路径下不包含数据文件。
        - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 参数。
        - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 参数或同时指定了 `sampler` 和 `shard_id` 参数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - `dataset_dir` 不存在。
        - **ValueError** - `usage` 不是["train", "test"]中的任何一个。
        - **ValueError** - `name` 不是["notredame", "yosemite", "liberty","notredame_harris", "yosemite_harris", "liberty_harris"]中的任何一个。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。
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

    **关于PhotoTour数据集：**

    数据取自许愿池（罗马）、巴黎圣母院（巴黎）和半圆顶（美国约塞米蒂国家公园）的旅游圣地照片。
    每个数据集包括一系列相应的图像块，是通过将旅游圣地的照片中的3D点投影回到原始图像而获得的。

    数据集由1024 x 1024位图（.bmp）图像组成，每个图像都包含16 x 16的图像修补数组。
    每个图像块都以64 x 64灰度采样，具有规范的比例和方向。有关如何确定比例和方向的详细信息，请参见论文。
    关联的元数据文件info.txt包含匹配信息。info.txt的每一行对应一个单独的图像块，图像块在每个位图图像中从左到右、从上到下顺序排列。
    info.txt每行上的第一个数字是采样该图像块的3D点ID——具有相同3D点ID的图像块从同一3D点投影（到不同的图像中）。
    info.txt中的第二个数字代表图像块是从哪个原始图像采样得到，目前未使用。

    可以将原始PhotoTour数据集文件解压缩到此目录结构中，并通过MindSpore的API读取。

    .. code-block::

        .
        └── photo_tour_dataset_directory
            ├── liberty/
            │    ├── info.txt                 // two columns: 3D_point_ID, unused
            │    ├── m50_100000_100000_0.txt  // seven columns: patch_ID1, 3D_point_ID1, unused1,
            │    │                            // patch_ID2, 3D_point_ID2, unused2, unused3
            │    ├── patches0000.bmp          // 1024*1024 pixels, with 16 * 16 patches.
            │    ├── patches0001.bmp
            │    ├── ...
            ├── yosemite/
            │    ├── ...
            ├── notredame/
            │    ├── ...
            ├── liberty_harris/
            │    ├── ...
            ├── yosemite_harris/
            │    ├── ...
            ├── notredame_harris/
            │    ├── ...

    **引用：**

    .. code-block::

        @INPROCEEDINGS{4269996,
            author={Winder, Simon A. J. and Brown, Matthew},
            booktitle={2007 IEEE Conference on Computer Vision and Pattern Recognition},
            title={Learning Local Image Descriptors},
            year={2007},
            volume={},
            number={},
            pages={1-8},
            doi={10.1109/CVPR.2007.382971}
        }


.. include:: mindspore.dataset.api_list_vision.rst
