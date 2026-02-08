mindspore.dataset.Flowers102Dataset
===================================

.. py:class:: mindspore.dataset.Flowers102Dataset(dataset_dir, task='Classification', usage='all', num_samples=None, num_parallel_workers=1, shuffle=None, decode=False, sampler=None, num_shards=None, shard_id=None)

    Oxford 102 Flower数据集。

    根据给定的 `task` 配置，生成数据集具有不同的输出列：

    - `task` 为 ``'Classification'`` ，输出列： `[image, dtype=uint8]` 、 `[label, dtype=uint32]` 。
    - `task` 为 ``'Segmentation'`` ，输出列： `[image, dtype=uint8]` 、 `[segmentation, dtype=uint8]` 、 `[label, dtype=uint32]` 。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **task** (str, 可选) - 指定读取数据的任务类型，支持 ``'Classification'`` 和 ``'Segmentation'``。默认值： ``'Classification'`` 。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 ``'train'`` 、 ``'valid'`` 、 ``'test'`` 或 ``'all'`` 。默认值： ``'all'`` ，读取全部样本。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值： ``None`` ，所有图像样本。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作进程数。默认值： ``1`` 。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值： ``None`` 。下表中会展示不同参数配置的预期行为。
        - **decode** (bool, 可选) - 是否对读取的图片进行解码操作。默认值： ``False`` ，不解码。
        - **sampler** (Union[Sampler, Iterable], 可选) - 指定从数据集中选取样本的采样器。默认值： ``None`` 。下表中会展示不同配置的预期行为。
        - **num_shards** (int, 可选) - 指定分布式训练时数据集划分的片数。默认值： ``None`` 。指定此参数后， `num_samples` 表示每个分片的最大样本数。一般在 `数据并行模式训练 <https://www.mindspore.cn/tutorials/zh-CN/master/parallel/data_parallel.html#数据集加载>`_ 的时候使用。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值： ``None`` 。只有当指定了 `num_shards` 时才能指定此参数。

    异常：
        - **RuntimeError** - `dataset_dir` 路径下不包含任何数据文件。
        - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 参数。
        - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 参数，或同时指定了 `sampler` 和 `shard_id` 参数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。
        - **ValueError** - 如果 `shard_id` 取值不在[0, `num_shards` )范围。

    教程样例：
        - `使用数据Pipeline加载 & 处理数据集
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/dataset_gallery.html>`_

    .. note:: 入参 `num_samples` 、 `shuffle` 、 `num_shards` 、 `shard_id` 可用于控制数据集所使用的采样器，其与入参 `sampler` 搭配使用的效果如下。

    .. include:: mindspore.dataset.sampler.rst

    **关于Flowers102数据集：**

    Flowers102数据集由102个花类别组成，每个类由40到258张图像组成，这些花常见于英国。

    以下是原始的Flowers102数据集结构。
    可以将数据集文件解压缩到此目录结构中，并通过MindSpore的API读取。

    .. code-block::

        .
        └── flowers102_dataset_dir
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
