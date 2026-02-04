mindspore.dataset.DIV2KDataset
==============================

.. py:class:: mindspore.dataset.DIV2KDataset(dataset_dir, usage="train", downgrade="bicubic", scale=2, num_samples=None, num_parallel_workers=None, shuffle=None, decode=None, sampler=None, num_shards=None, shard_id=None, cache=None)

    DIV2K（DIVerse 2K resolution image）数据集。

    生成的数据集有两列： `[hr_image, lr_image]` 。 `hr_image` 列和 `lr_image` 列的数据类型都为uint8。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **usage** (str, 可选) - 指定数据集的子集。可取值为 ``'train'`` 、 ``'valid'`` 或 ``'all'`` 。默认值： ``'train'`` 。
        - **downgrade** (str, 可选) - 指定数据集的下采样模式，可取值为 ``'bicubic'`` 、 ``'unknown'`` 、 ``'mild'`` 、 ``'difficult'`` 或 ``'wild'`` 。默认值： ``'bicubic'`` 。
        - **scale** (int, 可选) - 指定数据集的缩放尺度。当参数 `downgrade` 取值为 ``'bicubic'`` 时，此参数可以取值为 ``2`` 、 ``3`` 、 ``4`` 、``8`` 。
          当参数 `downgrade` 取值为 ``'unknown'`` 时，此参数可以取值为 ``2`` 、 ``3`` 、 ``4`` 。当参数 `downgrade` 取值为 ``'mild'`` 、 ``'difficult'`` 、 ``'wild'`` 时，此参数仅可以取值为 ``4`` 。默认值： ``2`` 。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数，可以小于数据集总数。默认值： ``None`` ，读取全部样本图片。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值： ``None`` ，使用全局默认线程数（8），也可以通过 :func:`mindspore.dataset.config.set_num_parallel_workers` 配置全局线程数。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值： ``None`` 。下表中会展示不同参数配置的预期行为。
        - **decode** (bool, 可选) - 是否对读取的图片进行解码操作。默认值： ``None`` ，相当于 ``False`` ，不解码。
        - **sampler** (Sampler, 可选) - 指定从数据集中选取样本的采样器。默认值： ``None`` 。下表中会展示不同配置的预期行为。
        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值： ``None`` 。指定此参数后， `num_samples` 表示每个分片的最大样本数。一般在 `数据并行模式训练 <https://www.mindspore.cn/tutorials/zh-CN/master/parallel/data_parallel.html#数据集加载>`_ 的时候使用。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值： ``None`` 。只有当指定了 `num_shards` 时才能指定此参数。
        - **cache** (:class:`~.dataset.DatasetCache`, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/zh-CN/master/dataset/cache.html>`_ 。默认值： ``None`` ，不使用缓存。

    异常：
        - **RuntimeError** - `dataset_dir` 路径下不包含任何数据文件。
        - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 参数。
        - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 参数，或同时指定了 `sampler` 和 `shard_id` 参数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - `dataset_dir` 路径非法或不存在。
        - **ValueError** - `usage` 参数取值不为 ``'train'`` 、 ``'valid'`` 或 ``'all'`` 。
        - **ValueError** - `downgrade` 参数取值不为 ``'bicubic'`` 、 ``'unknown'`` 、 ``'mild'`` 、 ``'difficult'`` 或 ``'wild'`` 。
        - **ValueError** - `scale` 参数取值不在允许的取值范围内，或与 `downgrade` 参数的值不匹配。
        - **ValueError** - `scale` 参数取值为 ``8`` ，但 `downgrade` 参数的值不为 ``'bicubic'`` 。
        - **ValueError** - `downgrade` 参数取值为 ``'mild'`` 、 ``'difficult'`` 或 ``'wild'`` ，但 `scale` 参数的值不为 ``4`` 。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。        
        - **ValueError** - 如果 `shard_id` 取值不在[0, `num_shards` )范围。

    教程样例：
        - `使用数据Pipeline加载 & 处理数据集
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/dataset_gallery.html>`_

    .. note:: 入参 `num_samples` 、 `shuffle` 、 `num_shards` 、 `shard_id` 可用于控制数据集所使用的采样器，其与入参 `sampler` 搭配使用的效果如下。

    .. include:: mindspore.dataset.sampler.rst

    **关于DIV2K数据集：**

    DIV2K数据集由1000张2K分辨率图像组成，其中800张用于训练、100张用于验证、100张用于测试。
    作为NTIRE比赛的数据集，NTIRE 2017 和 NTIRE 2018 仅包括DIV2K的训练数据集和验证数据集。

    您可以解压缩原始DIV2K数据集文件到如下目录结构，并通过MindSpore的API进行读取。

    以训练数据集作为例子。

    .. code-block::

        .
        └── DIV2K
             ├── DIV2K_train_HR
             |    ├── 0001.png
             |    ├── 0002.png
             |    ├── ...
             ├── DIV2K_train_LR_bicubic
             |    ├── X2
             |    |    ├── 0001x2.png
             |    |    ├── 0002x2.png
             |    |    ├── ...
             |    ├── X3
             |    |    ├── 0001x3.png
             |    |    ├── 0002x3.png
             |    |    ├── ...
             |    └── X4
             |         ├── 0001x4.png
             |         ├── 0002x4.png
             |         ├── ...
             ├── DIV2K_train_LR_unknown
             |    ├── X2
             |    |    ├── 0001x2.png
             |    |    ├── 0002x2.png
             |    |    ├── ...
             |    ├── X3
             |    |    ├── 0001x3.png
             |    |    ├── 0002x3.png
             |    |    ├── ...
             |    └── X4
             |         ├── 0001x4.png
             |         ├── 0002x4.png
             |         ├── ...
             ├── DIV2K_train_LR_mild
             |    ├── 0001x4m.png
             |    ├── 0002x4m.png
             |    ├── ...
             ├── DIV2K_train_LR_difficult
             |    ├── 0001x4d.png
             |    ├── 0002x4d.png
             |    ├── ...
             ├── DIV2K_train_LR_wild
             |    ├── 0001x4w.png
             |    ├── 0002x4w.png
             |    ├── ...
             └── DIV2K_train_LR_x8
                  ├── 0001x8.png
                  ├── 0002x8.png
                  ├── ...

    **引用：**

    .. code-block::

        @InProceedings{Agustsson_2017_CVPR_Workshops,
        author    = {Agustsson, Eirikur and Timofte, Radu},
        title     = {NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study},
        booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
        url       = "http://www.vision.ee.ethz.ch/~timofter/publications/Agustsson-CVPRW-2017.pdf",
        month     = {July},
        year      = {2017}
        }


.. include:: mindspore.dataset.api_list_vision.rst
