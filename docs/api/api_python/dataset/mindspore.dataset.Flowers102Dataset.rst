mindspore.dataset.Flowers102Dataset
===================================

.. py:class:: mindspore.dataset.Flowers102Dataset(dataset_dir, task='Classification', usage='all', num_samples=None, num_parallel_workers=1, shuffle=None, decode=False, sampler=None, num_shards=None, shard_id=None)

    读取和解析Flowers102数据集的源文件构建数据集。

    根据给定的 `task` 配置，生成数据集具有不同的输出列：

    - `task` = 'Classification'，输出列： `[image, dtype=uint8]` 、 `[label, dtype=uint32]` 。
    - `task` = 'Segmentation'，输出列： `[image, dtype=uint8]` 、 `[segmentation, dtype=uint8]` 、 `[label, dtype=uint32]` 。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录的路径。
        - **task** (str, 可选) - 指定读取数据的任务类型，支持 'Classification' 和 'Segmentation'。默认值：'Classification'。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 'train'、'valid'、'test' 或 'all'。默认值：'all'，读取全部样本。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None，所有图像样本。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：1。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值：None。下表中会展示不同参数配置的预期行为。
        - **decode** (bool, 可选) - 是否对读取的图片进行解码操作。默认值：False，不解码。
        - **sampler** (Union[Sampler, Iterable], 可选) - 指定从数据集中选取样本的采样器。默认值：None。下表中会展示不同配置的预期行为。
        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值：None。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值：None。只有当指定了 `num_shards` 时才能指定此参数。

    异常：
        - **RuntimeError** - `dataset_dir` 路径下不包含任何数据文件。
        - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 参数。
        - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 参数或同时指定了 `sampler` 和 `shard_id` 参数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
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

    **关于Flowers102数据集：**

    Flowers102数据集由102个花类别组成，每个类由40到258张图像组成，这些花常见于英国。

    以下是原始的Flowers102数据集结构。
    可以将数据集文件解压缩到此目录结构中，并通过MindSpore的API读取。

    .. code-block::

        .
        └── flowes102_dataset_dir
             ├── imagelabels.mat
             ├── setid.mat
             ├── jpg
                  ├── image_00001.jpg
                  ├── image_00002.jpg
                  ├── ...
             ├── segmim
                  ├── segmim_00001.jpg
                  ├── segmim_00002.jpg
                  ├── ...

    **引用：**

    .. code-block::

        @InProceedings{Nilsback08,
          author       = "Maria-Elena Nilsback and Andrew Zisserman",
          title        = "Automated Flower Classification over a Large Number of Classes",
          booktitle    = "Indian Conference on Computer Vision, Graphics and Image Processing",
          month        = "Dec",
          year         = "2008",
        }


.. include:: mindspore.dataset.api_list_vision.rst
