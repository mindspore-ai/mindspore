mindspore.dataset.CMUArcticDataset
===================================

.. py:class:: mindspore.dataset.CMUArcticDataset(dataset_dir, name=None, num_samples=None, num_parallel_workers=None, shuffle=None, sampler=None, num_shards=None, shard_id=None, cache=None)

    CMU Arctic数据集。

    生成的数据集有四列 `[waveform, sample_rate, transcript, utterance_id]` 。
    `waveform` 列的数据类型为float32。
    `sample_rate` 列的数据类型为uint32。
    `transcript` 列的数据类型为string。
    `utterance_id` 列的数据类型为string。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **name** (str, 可选) - 指定读取的数据集子集，可为 ``'aew'`` 、 ``'ahw'`` 、 ``'aup'`` 、 ``'awb'`` 、 ``'axb'`` 、 ``'bdl'`` 、
          ``'clb'`` 、 ``'eey'`` 、 ``'fem'`` 、 ``'gka'``、 ``'jmk'``、 ``'ksp'``、 ``'ljm'``、 ``'lnh'``、 ``'rms'``、 ``'rxr'``、 ``'slp'`` 或 ``'slt'``。默认值： ``None`` ，表示 ``'aew'`` 。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值： ``None`` ，读取全部音频。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值： ``None`` ，使用全局默认线程数（8），也可以通过 :func:`mindspore.dataset.config.set_num_parallel_workers` 配置全局线程数。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值： ``None`` 。下表中会展示不同参数配置的预期行为。
        - **sampler** (Sampler, 可选) - 指定从数据集中选取样本的采样器。默认值： ``None`` 。下表中会展示不同配置的预期行为。
        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值： ``None`` ，不进行分片。指定此参数后， `num_samples` 表示每个分片的最大样本数。一般在 `数据并行模式训练 <https://www.mindspore.cn/tutorials/zh-CN/master/parallel/data_parallel.html#数据集加载>`_ 的时候使用。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值： ``None`` ，将使用 ``0`` 。只有当指定了 `num_shards` 时才能指定此参数。
        - **cache** (:class:`~.dataset.DatasetCache`, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/zh-CN/master/dataset/cache.html>`_ 。默认值： ``None`` ，不使用缓存。

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

    .. note::
        - 暂不支持指定 `sampler` 参数为 :class:`mindspore.dataset.PKSampler`。
        - 入参 `num_samples` 、 `shuffle` 、 `num_shards` 、 `shard_id` 可用于控制数据集所使用的采样器，其与入参 `sampler` 搭配使用的效果如下。

    .. include:: mindspore.dataset.sampler.rst

    **关于CMUArctic数据集：**

    CMU Arctic数据集是为语音合成研究而设计的。这些单人语音数据是在演播室条件下精心录制的，由大约1200个\
    语音平衡的英语语料组成。除了音频文件外，数据集还为Festival语音合成系统提供了完整的支持，包括可按原\
    样使用的预建语音。整个软件包是作为免费软件发布的，不限制商业或非商业使用。

    您可以将CMUArctic数据集构建成以下目录结构，并通过MindSpore的API进行读取。

    .. code-block::

        .
        └── cmu_arctic_dataset_directory
            ├── cmu_us_aew_arctic
            │    ├── wav
            │    │    ├──arctic_a0001.wav
            │    │    ├──arctic_a0002.wav
            │    │    ├──...
            │    ├── etc
            │    │    └── txt.done.data
            ├── cmu_us_ahw_arctic
            │    ├── wav
            │    │    ├──arctic_a0001.wav
            │    │    ├──arctic_a0002.wav
            │    │    ├──...
            │    └── etc
            │         └── txt.done.data
            └──...

    **引用：**

    .. code-block::

        @article{LTI2003CMUArctic,
        title        = {CMU ARCTIC databases for speech synthesis},
        author       = {John Kominek and Alan W Black},
        journal      = {Language Technologies Institute [Online]},
        year         = {2003}
        howpublished = {http://www.festvox.org/cmu_arctic/}
        }


.. include:: mindspore.dataset.api_list_audio.rst
