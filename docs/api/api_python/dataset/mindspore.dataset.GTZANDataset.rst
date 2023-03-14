mindspore.dataset.GTZANDataset
===============================

.. py:class:: mindspore.dataset.GTZANDataset(dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None, sampler=None, num_shards=None, shard_id=None, cache=None)

    读取和解析GTZAN数据集的源文件构建数据集。

    生成的数据集有三列 `[waveform, sample_rate, label]` 。
    `waveform` 列的数据类型为float32。
    `sample_rate` 列的数据类型为uint32。
    `label` 列的数据类型为string。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 'train'，'valid'，'test' 或 'all'。默认值：None，表示读取全部样本。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None，读取全部音频。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用mindspore.dataset.config中配置的线程数。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值：None，下表中会展示不同参数配置的预期行为。
        - **sampler** (Sampler, 可选) - 指定从数据集中选取样本的采样器。默认值：None，下表中会展示不同配置的预期行为。
        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值：None。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值：None。只有当指定了 `num_shards` 时才能指定此参数。
        - **cache** (DatasetCache, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/cache.html>`_ 。默认值：None，不使用缓存。

    异常：
        - **RuntimeError** - `dataset_dir` 路径下不包含任何数据文件。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。
        - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 参数。
        - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 参数或同时指定了 `sampler` 和 `shard_id` 参数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - `shard_id` 参数错误，小于0或者大于等于 `num_shards` 。

    .. note::
        - GTZANDataset的 `sampler` 参数不支持指定PKSampler。
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

    **关于GTZAN数据集：**

    GTZAN数据集至少出现在100个已发表的工作中，是机器听觉研究中最常用的公共数据集，用于音乐曲风识别。
    它由1000条音轨组成，每条音轨的长度为30秒。它包含10种流派（蓝调、古典、乡村、迪斯科、hiphop、爵士、
    金属、流行、雷鬼和摇滚），每一种曲风各100条音轨。这些音轨都是22050Hz的单声道16位.wav格式音频文件。

    您可以将GTZAN数据集构建成以下目录结构，并通过MindSpore的API进行读取。

    .. code-block::

        .
        └── gtzan_dataset_directory
            ├── blues
            │    ├──blues.00000.wav
            │    ├──blues.00001.wav
            │    ├──blues.00002.wav
            │    ├──...
            ├── disco
            │    ├──disco.00000.wav
            │    ├──disco.00001.wav
            │    ├──disco.00002.wav
            │    └──...
            └──...

    **引用：**

    .. code-block::

        @misc{tzanetakis_essl_cook_2001,
        author    = "Tzanetakis, George and Essl, Georg and Cook, Perry",
        title     = "Automatic Musical Genre Classification Of Audio Signals",
        url       = "http://ismir2001.ismir.net/pdf/tzanetakis.pdf",
        publisher = "The International Society for Music Information Retrieval",
        year      = "2001"
        }


.. include:: mindspore.dataset.api_list_audio.rst
