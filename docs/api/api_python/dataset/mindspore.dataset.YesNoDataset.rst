mindspore.dataset.YesNoDataset
==============================

.. py:class:: mindspore.dataset.YesNoDataset(dataset_dir, num_samples=None, num_parallel_workers=None, shuffle=None, sampler=None, num_shards=None, shard_id=None, cache=None)

    YesNo数据集。

    生成的数据集有三列 `[waveform, sample_rate, labels]` 。
    列 `waveform` 的数据类型为float32。列 `sample_rate` 的数据类型为int32。列 `labels` 的数据类型为int32。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值： ``None`` ，读取全部样本。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值： ``None`` ，使用全局默认线程数(8)，也可以通过 :func:`mindspore.dataset.config.set_num_parallel_workers` 配置全局线程数。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值： ``None`` 。下表中会展示不同参数配置的预期行为。
        - **sampler** (Sampler, 可选) - 指定从数据集中选取样本的采样器。默认值： ``None`` 。下表中会展示不同配置的预期行为。
        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值： ``None`` 。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值： ``None`` 。只有当指定了 `num_shards` 时才能指定此参数。
        - **cache** (:class:`~.dataset.DatasetCache`, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/cache.html>`_ 。默认值： ``None`` ，不使用缓存。

    异常：
        - **RuntimeError** - `dataset_dir` 路径下不包含任何数据文件。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。
        - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 参数。
        - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 参数或同时指定了 `sampler` 和 `shard_id` 参数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - 如果 `shard_id` 取值不在[0, `num_shards` )范围。

    教程样例：
        - `使用数据Pipeline加载 & 处理数据集
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/dataset_gallery.html>`_

    .. note:: 入参 `num_samples` 、 `shuffle` 、 `num_shards` 、 `shard_id` 可用于控制数据集所使用的采样器，其与入参 `sampler` 搭配使用的效果如下。

    .. include:: mindspore.dataset.sampler.rst

    **关于YesNo数据集：**

    Yesno是一个音频数据集，由60个录音组成，由一个人用希伯来语说是或不是；每个录音都有8个字长。

    以下是原始的YesNo数据集结构。可以将数据集文件解压缩到此目录结构中，并由MindSpore的API读取。

    .. code-block::

        .
        └── yes_no_dataset_dir
             ├── 1_1_0_0_1_1_0_0.wav
             ├── 1_0_0_0_1_1_0_0.wav
             ├── 1_1_0_0_1_1_0_0.wav
             └──....

    **引用：**

    .. code-block::

        @NetworkResource{Kaldi_audio_project,
        author    = {anonymous},
        url       = "http://wwww.openslr.org/1/"
        }


.. include:: mindspore.dataset.api_list_audio.rst
