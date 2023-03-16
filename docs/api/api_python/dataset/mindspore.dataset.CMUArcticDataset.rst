mindspore.dataset.CMUArcticDataset
===================================

.. py:class:: mindspore.dataset.CMUArcticDataset(dataset_dir, name=None, num_samples=None, num_parallel_workers=None, shuffle=None, sampler=None, num_shards=None, shard_id=None, cache=None)

    读取和解析CMUArctic数据集的源文件构建数据集。

    生成的数据集有四列 `[waveform, sample_rate, transcript, utterance_id]` 。
    `waveform` 列的数据类型为float32。
    `sample_rate` 列的数据类型为uint32。
    `transcript` 列的数据类型为string。
    `utterance_id` 列的数据类型为string。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **name** (str, 可选) - 指定读取的数据集子集，可为 'aew'、'ahw'、'aup'、'awb'、'axb'、'bdl'、
          'clb'、'eey'、'fem'、'gka'、'jmk'、'ksp'、'ljm'、'lnh'、'rms'、'rxr'、'slp' 或 'slt'。默认值：None，表示 'aew'。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None，读取全部音频。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用 `mindspore.dataset.config` 中配置的线程数。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值：None，下表中会展示不同参数配置的预期行为。
        - **sampler** (Sampler, 可选) - 指定从数据集中选取样本的采样器。默认值：None，下表中会展示不同配置的预期行为。
        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值：None，不进行分片。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值：None，将使用0。只有当指定了 `num_shards` 时才能指定此参数。
        - **cache** (DatasetCache, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/cache.html>`_ 。默认值：None，不使用缓存。

    异常：
        - **RuntimeError** - `dataset_dir` 路径下不包含任何数据文件。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。
        - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 参数。
        - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 参数或同时指定了 `sampler` 和 `shard_id` 参数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - 如果 `shard_id` 取值不在[0, `num_shards` )范围。

    .. note::
        - 暂不支持指定 `sampler` 参数为 `mindspore.dataset.PKSampler`。
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

    **关于CMUArctic数据集：**

    CMU Arctic数据集是为语音合成研究而设计的。这些单人语音数据是在演播室条件下精心录制的，由大约1200个
    语音平衡的英语语料组成。除了音频文件外，数据集还为Festival语音合成系统提供了完整的支持，包括可按原
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
