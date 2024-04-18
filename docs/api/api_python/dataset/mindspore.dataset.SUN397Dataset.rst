mindspore.dataset.SUN397Dataset
===============================

.. py:class:: mindspore.dataset.SUN397Dataset(dataset_dir, num_samples=None, num_parallel_workers=None, shuffle=None, decode=False, sampler=None, num_shards=None, shard_id=None, cache=None)

    SUN397（Scene UNderstanding）数据集。

    生成的数据集有两列：`[image, label]`。`image` 列的数据类型是uint8。`label` 列的数据类型是uint32。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数，可以小于数据集总数。默认值： ``None`` ，读取全部样本图片。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值： ``None`` ，使用全局默认线程数(8)，也可以通过 :func:`mindspore.dataset.config.set_num_parallel_workers` 配置全局线程数。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值： ``None`` ，下表中会展示不同参数配置的预期行为。
        - **decode** (bool, 可选) - 是否对读取的图片进行解码操作。默认值： ``False`` ，不解码。
        - **sampler** (Sampler, 可选) - 指定从数据集中选取样本的采样器。默认值： ``None`` ，下表中会展示不同配置的预期行为。
        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值： ``None`` 。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值： ``None`` 。只有当指定了 `num_shards` 时才能指定此参数。
        - **cache** (:class:`~.dataset.DatasetCache`, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/cache.html>`_ 。默认值： ``None`` ，不使用缓存。

    异常：
        - **RuntimeError** - `dataset_dir` 路径下不包含数据文件。
        - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 参数。
        - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 参数或同时指定了 `sampler` 和 `shard_id` 参数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。
        - **ValueError** - 如果 `shard_id` 取值不在[0, `num_shards` )范围。

    教程样例：
        - `使用数据Pipeline加载 & 处理数据集
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/dataset_gallery.html>`_

    .. note:: 入参 `num_samples` 、 `shuffle` 、 `num_shards` 、 `shard_id` 可用于控制数据集所使用的采样器，其与入参 `sampler` 搭配使用的效果如下。

    .. include:: mindspore.dataset.sampler.rst

    **关于SUN397数据集：**

    SUN397是一个用于场景识别的数据集，包括397个类别，有108,754张图像。不同类别的图像数量不同，但每个类别至少有100张。
    图片为jpg、png或gif格式。

    以下是原始SUN397数据集结构。
    可以将数据集文件解压缩到此目录结构中，并由MindSpore的API读取。

    .. code-block::

        .
        └── sun397_dataset_directory
            ├── ClassName.txt
            ├── README.txt
            ├── a
            │   ├── abbey
            │   │   ├── sun_aaaulhwrhqgejnyt.jpg
            │   │   ├── sun_aacphuqehdodwawg.jpg
            │   │   ├── ...
            │   ├── apartment_building
            │   │   └── outdoor
            │   │       ├── sun_aamyhslnsnomjzue.jpg
            │   │       ├── sun_abbjzfrsalhqivis.jpg
            │   │       ├── ...
            │   ├── ...
            ├── b
            │   ├── badlands
            │   │   ├── sun_aabtemlmesogqbbp.jpg
            │   │   ├── sun_afbsfeexggdhzshd.jpg
            │   │   ├── ...
            │   ├── balcony
            │   │   ├── exterior
            │   │   │   ├── sun_aaxzaiuznwquburq.jpg
            │   │   │   ├── sun_baajuldidvlcyzhv.jpg
            │   │   │   ├── ...
            │   │   └── interior
            │   │       ├── sun_babkzjntjfarengi.jpg
            │   │       ├── sun_bagjvjynskmonnbv.jpg
            │   │       ├── ...
            │   └── ...
            ├── ...

    **引用：**

    .. code-block::

        @inproceedings{xiao2010sun,
        title        = {Sun database: Large-scale scene recognition from abbey to zoo},
        author       = {Xiao, Jianxiong and Hays, James and Ehinger, Krista A and Oliva, Aude and Torralba, Antonio},
        booktitle    = {2010 IEEE computer society conference on computer vision and pattern recognition},
        pages        = {3485--3492},
        year         = {2010},
        organization = {IEEE}
        }


.. include:: mindspore.dataset.api_list_vision.rst
