mindspore.dataset.LJSpeechDataset
=================================

.. py:class:: mindspore.dataset.LJSpeechDataset(dataset_dir, num_samples=None, num_parallel_workers=None, shuffle=None, sampler=None, num_shards=None, shard_id=None, cache=None)

    读取和解析LJSpeech数据集的源文件构建数据集。

    生成的数据集有四列: `[waveform, sample_rate, transcription, normalized_transcript]` 。
    `waveform` 列的数据类型为float32。 `sample_rate` 列的数据类型为int32。 `transcription` 列的数据类型为string。 `normalized_transcript` 列的数据类型为string。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数，可以小于数据集总数。默认值：None，读取全部样本音频。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用 `mindspore.dataset.config` 中配置的线程数。
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

    **关于LJSPEECH数据集：**
    
    LJSPEECH是一个公共领域的语音数据集，由13,100个来自7部非小说类书籍的段落短音频片段组成。
    为每个剪辑片段都进行转录。剪辑的长度从1秒到10秒不等，总长度约为24小时。

    这些被阅读的文本于1884年至1964年间出版，属于公共领域。这些音频由LibriVox项目于2016-17年录制。

    以下是原始的LJSPEECH数据集结构。
    可以将数据集文件解压缩到以下目录结构中，并由MindSpore的API读取。

    .. code-block::

        .
        └── LJSpeech-1.1
            ├── README
            ├── metadata.csv
            └── wavs
                ├── LJ001-0001.wav
                ├── LJ001-0002.wav
                ├── LJ001-0003.wav
                ├── LJ001-0004.wav
                ├── LJ001-0005.wav
                ├── LJ001-0006.wav
                ├── LJ001-0007.wav
                ├── LJ001-0008.wav
                ...
                ├── LJ050-0277.wav
                └── LJ050-0278.wav

    **引用：**

    .. code-block::

        @misc{lj_speech17,
        author       = {Keith Ito and Linda Johnson},
        title        = {The LJ Speech Dataset},
        howpublished = {url{https://keithito.com/LJ-Speech-Dataset}},
        year         = 2017
        }


.. include:: mindspore.dataset.api_list_audio.rst
