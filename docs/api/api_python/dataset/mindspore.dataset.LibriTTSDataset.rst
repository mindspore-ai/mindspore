mindspore.dataset.LibriTTSDataset
=================================

.. py:class:: mindspore.dataset.LibriTTSDataset(dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None, sampler=None, num_shards=None, shard_id=None, cache=None)

    LibriTTS数据集。

    生成的数据集有七列 `[waveform, sample_rate, original_text, normalized_text, speaker_id, chapter_id, utterance_id]` 。
    `waveform` 列的数据类型为float32。
    `sample_rate` 列的数据类型为uint32。
    `original_text` 列的数据类型为string。
    `normalized_text` 列的数据类型为string。
    `speaker_id` 列的数据类型为uint32。
    `chapter_id` 列的数据类型为uint32。
    `utterance_id` 列的数据类型为string。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 'dev-clean'、'dev-other'、'test-clean'、'test-other'、'train-clean-100'、'train-clean-360'、'train-other-500' 或 'all'。默认值：None，表示读取全部样本。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None，读取全部音频。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用全局默认线程数(8)，也可以通过 `mindspore.dataset.config.set_num_parallel_workers` 配置全局线程数。
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

    **关于LibriTTS数据集：**

    LibriTTS是由Heiga Zen在谷歌语音和谷歌大脑团队成员的协助下准备的一个多语言英语语料库，
    包括了大约585小时的24kHz采样率的英语语音。LibriTTS语料库是为TTS研究设计的。它是由
    LibriSpeech语料库的原始语料（来自LibriVox的mp3音频文件和来自Project Gutenberg的文本文件）
    衍生而来。

    您可以将LibriTTS数据集构建成以下目录结构，并通过MindSpore的API进行读取。

    .. code-block::

        .
        └── libri_tts_dataset_directory
            ├── dev-clean
            │    ├── 116
            │    │    ├── 288045
            |    |    |    ├── 116_288045.trans.tsv
            │    │    │    ├── 116_288045_000003_000000.wav
            │    │    │    └──...
            │    │    ├── 288046
            |    |    |    ├── 116_288046.trans.tsv
            |    |    |    ├── 116_288046_000003_000000.wav
            │    |    |    └── ...
            |    |    └── ...
            │    ├── 1255
            │    │    ├── 138279
            |    |    |    ├── 1255_138279.trans.tsv
            │    │    │    ├── 1255_138279_000001_000000.wav
            │    │    │    └── ...
            │    │    ├── 74899
            |    |    |    ├── 1255_74899.trans.tsv
            |    |    |    ├── 1255_74899_000001_000000.wav
            │    |    |    └── ...
            |    |    └── ...
            |    └── ...
            └── ...

    **引用：**

    .. code-block::

        @article{lecun2010mnist,
        title        = {LIBRITTS handwritten digit database},
        author       = {zpw, NBU},
        journal      = {ATT Labs [Online]},
        volume       = {2},
        year         = {2010},
        howpublished = {http://www.openslr.org/resources/60/},
        description  = {The LibriSpeech ASR corpus (http://www.openslr.org/12/) [1] has been used in
                        various research projects. However, as it was originally designed for ASR research,
                        there are some undesired properties when using for TTS research}
        }


.. include:: mindspore.dataset.api_list_audio.rst
