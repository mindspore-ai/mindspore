mindspore.dataset.CityscapesDataset
===================================

.. py:class:: mindspore.dataset.CityscapesDataset(dataset_dir, usage="train", quality_mode="fine", task="instance", num_samples=None, num_parallel_workers=None, shuffle=None, decode=None, sampler=None, num_shards=None, shard_id=None, cache=None)

    Cityscapes数据集。

    生成的数据集有两列 `[image, task]` 。
    `image` 列的数据类型为uint8。`task` 列的数据类型根据参数 `task` 的值而定。当参数 `task` 取值为 ``'polygon'`` 时，列的数据类型为string；其他取值下，列的数据类型为uint8。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **usage** (str, 可选) - 指定数据集的子集。当参数 `quality_mode` 取值为 ``'fine'`` 时，此参数可取值为 ``'train'`` 、 ``'test'`` 、 ``'val'`` 或 ``'all'`` 。
          当参数 `quality_mode` 取值为 ``'coarse'`` 时，此参数可取值为 ``'train'`` 、 ``'train_extra'`` 、 ``'val'`` 或 ``'all'`` 。默认值： ``'train'`` ，读取训练集样本。
        - **quality_mode** (str, 可选) - 指定数据集的质量模式，可取值为 ``'fine'`` 或 ``'coarse'`` 。默认值： ``'fine'`` 。
        - **task** (str, 可选) - 指定数据集的任务类型，可取值为 ``'instance'`` 、 ``'semantic'`` 、 ``'polygon'`` 或 ``'color'`` 。默认值： ``'instance'`` 。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数，可以小于数据集总数。默认值： ``None`` ，读取全部样本图片。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值： ``None`` ，使用全局默认线程数（8），也可以通过 :func:`mindspore.dataset.config.set_num_parallel_workers` 配置全局线程数。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值： ``None`` 。下表中会展示不同参数配置的预期行为。
        - **decode** (bool, 可选) - 是否对读取的图片进行解码操作。默认值： ``None`` ，表示 ``False`` ，不解码。
        - **sampler** (Sampler, 可选) - 指定从数据集中选取样本的采样器。默认值： ``None`` 。下表中会展示不同配置的预期行为。
        - **num_shards** (int, 可选) - 指定分布式训练时，数据集划分的片数。默认值： ``None`` 。指定此参数后， `num_samples` 表示每个分片的最大样本数。一般在 `数据并行模式训练 <https://www.mindspore.cn/tutorials/zh-CN/master/parallel/data_parallel.html#数据集加载>`_ 的时候使用。
        - **shard_id** (int, 可选) - 指定分布式训练时，使用的分片ID号。默认值： ``None`` 。只有当指定了 `num_shards` 时才能指定此参数。
        - **cache** (:class:`~.dataset.DatasetCache`, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/zh-CN/master/dataset/cache.html>`_ 。默认值： ``None`` ，不使用缓存。

    教程样例：
        - `使用数据Pipeline加载 & 处理数据集
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/dataset_gallery.html>`_

    异常：
        - **RuntimeError** - `dataset_dir` 路径下不包含任何数据文件。
        - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 参数。
        - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 参数，或同时指定了 `sampler` 和 `shard_id` 参数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。
        - **ValueError** - `dataset_dir` 路径非法或不存在。
        - **ValueError** - `task` 参数取值不为 ``'instance'`` 、 ``'semantic'``、 ``'polygon'`` 或 ``'color'`` 。
        - **ValueError** - `quality_mode` 参数取值不为 ``'fine'`` 或 ``'coarse'`` 。
        - **ValueError** - `usage` 参数取值不在给定的字段中。
        - **ValueError** - 如果 `shard_id` 取值不在[0, `num_shards` )范围。

    .. note:: 入参 `num_samples` 、 `shuffle` 、 `num_shards` 、 `shard_id` 可用于控制数据集所使用的采样器，其与入参 `sampler` 搭配使用的效果如下。

    .. include:: mindspore.dataset.sampler.rst

    **关于Cityscapes数据集：**

    Cityscapes 数据集由来自 50 个城市的 24998 张彩色图像组成。
    其中 5000 张图像具有高质量的密集像素标注，19998 张图像具有粗糙的多边形标注。
    该数据集共有 30 个类，多边形标注包括密集语义分割，以及车辆和人的实例分割。

    您可以解压缩原始数据集文件到如下目录结构，并通过MindSpore的API进行读取。

    .. code-block::

        .
        └── Cityscapes
             ├── leftImg8bit
             |    ├── train
             |    |    ├── aachen
             |    |    |    ├── aachen_000000_000019_leftImg8bit.png
             |    |    |    ├── aachen_000001_000019_leftImg8bit.png
             |    |    |    ├── ...
             |    |    ├── bochum
             |    |    |    ├── ...
             |    |    ├── ...
             |    ├── test
             |    |    ├── ...
             |    ├── val
             |    |    ├── ...
             └── gtFine
                  ├── train
                  |    ├── aachen
                  |    |    ├── aachen_000000_000019_gtFine_color.png
                  |    |    ├── aachen_000000_000019_gtFine_instanceIds.png
                  |    |    ├── aachen_000000_000019_gtFine_labelIds.png
                  |    |    ├── aachen_000000_000019_gtFine_polygons.json
                  |    |    ├── aachen_000001_000019_gtFine_color.png
                  |    |    ├── aachen_000001_000019_gtFine_instanceIds.png
                  |    |    ├── aachen_000001_000019_gtFine_labelIds.png
                  |    |    ├── aachen_000001_000019_gtFine_polygons.json
                  |    |    ├── ...
                  |    ├── bochum
                  |    |    ├── ...
                  |    ├── ...
                  ├── test
                  |    ├── ...
                  └── val
                       ├── ...

    **引用：**

    .. code-block::

        @inproceedings{Cordts2016Cityscapes,
        title       = {The Cityscapes Dataset for Semantic Urban Scene Understanding},
        author      = {Cordts, Marius and Omran, Mohamed and Ramos, Sebastian and Rehfeld, Timo and Enzweiler,
                        Markus and Benenson, Rodrigo and Franke, Uwe and Roth, Stefan and Schiele, Bernt},
        booktitle   = {Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year        = {2016}
        }


.. include:: mindspore.dataset.api_list_vision.rst
