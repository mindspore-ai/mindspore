mindspore.dataset.TedliumDataset
================================

.. py:class:: mindspore.dataset.TedliumDataset(dataset_dir, release, usage=None, extensions=None, num_samples=None, num_parallel_workers=None, shuffle=None, sampler=None, num_shards=None, shard_id=None, cache=None)

    Tedlium数据集。生成的数据集的列取决于源SPH文件和相应的STM文件。

    生成的数据集有六列 `[waveform, sample_rate, transcript, talk_id, speaker_id, identifier]`。
    列 `waveform` 的数据类型为float32，列 `sample_rate` 的数据类型为int32，列 `transcript`、列 `talk_id`、列 `speaker_id` 和列 `identifier` 的数据类型为string。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **release** (str) - 指定数据集的发布版本，可以取值为 'release1'、'release2' 或 'release3'。
        - **usage** (str, 可选) - 指定数据集的子集。
          对于 `release` 为 'release1' 或 'release2'，`usage` 可以是 'train'、'test'、'dev' 或 'all'。
          对于 `release` 为 'release3'， `usage` 只能是 'all'。默认值：None，读取全部样本。
        - **extensions** (str, 可选) - 指定SPH文件的扩展名。默认值：'.sph'。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None，读取全部样本。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用全局默认线程数(8)，也可以通过 `mindspore.dataset.config.set_num_parallel_workers` 配置全局线程数。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值：None。下表中会展示不同参数配置的预期行为。
        - **sampler** (Sampler, 可选) - 指定从数据集中选取样本的采样器。默认值：None。下表中会展示不同配置的预期行为。
        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值：None。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值：None。只有当指定了 `num_shards` 时才能指定此参数。
        - **cache** (DatasetCache, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/cache.html>`_ 。默认值：None，不使用缓存。

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

    **关于TEDLIUM数据集：**

    TEDLIUM_release1数据集：TED-LUM语料库是英语TED演讲，有转录，采样频率为16kHz。包含了大约118小时的演讲。

    TEDLIUM_release2数据集：这是TED-LIUM语料库版本2，根据知识共享BY-NC-ND 3.0授权。所有会谈和文本均为TED会议有限责任公司的财产。TED-LIUM语料库是由音频谈话和他们的转录在TED网站上提供的。我们准备并过滤了这些数据，以便训练声学模型参加2011年口语翻译国际研讨会（LIUM英语/法语SLT系统在SLT任务中排名第一）。

    TEDLIUM_release-3数据集：这是TED-LIUM语料库版本3，根据知识共享BY-NC-ND 3.0授权。所有会谈和文本均为TED会议有限责任公司的财产。这个新的TED-LIUM版本是通过Ubiqus公司和LIUM（法国勒芒大学）的合作发布的。

    可以将数据集文件解压缩到以下目录结构中，并由MindSpore的API读取。

    TEDLIUM release1与TEDLIUM release2的结构相同，只是数据不同。

    .. code-block::

        .
        └──TEDLIUM_release1
            └── dev
                ├── sph
                    ├── AlGore_2009.sph
                    ├── BarrySchwartz_2005G.sph
                ├── stm
                    ├── AlGore_2009.stm
                    ├── BarrySchwartz_2005G.stm
            └── test
                ├── sph
                    ├── AimeeMullins_2009P.sph
                    ├── BillGates_2010.sph
                ├── stm
                    ├── AimeeMullins_2009P.stm
                    ├── BillGates_2010.stm
            └── train
                ├── sph
                    ├── AaronHuey_2010X.sph
                    ├── AdamGrosser_2007.sph
                ├── stm
                    ├── AaronHuey_2010X.stm
                    ├── AdamGrosser_2007.stm
            └── readme
            └── TEDLIUM.150k.dic

    TEDLIUM release3目录结构稍有不同。

    .. code-block::

        .
        └──TEDLIUM_release-3
            └── data
                ├── ctl
                ├── sph
                    ├── 911Mothers_2010W.sph
                    ├── AalaElKhani.sph
                ├── stm
                    ├── 911Mothers_2010W.stm
                    ├── AalaElKhani.stm
            └── doc
            └── legacy
            └── LM
            └── speaker-adaptation
            └── readme
            └── TEDLIUM.150k.dic

    **引用：**

    .. code-block::

        @article{
          title={TED-LIUM: an automatic speech recognition dedicated corpus},
          author={A. Rousseau, P. Deléglise, Y. Estève},
          journal={Proceedings of the Eighth International Conference on Language Resources and Evaluation (LREC'12)},
          year={May 2012},
          biburl={https://www.openslr.org/7/}
        }

        @article{
          title={Enhancing the TED-LIUM Corpus with Selected Data for Language Modeling and More TED Talks},
          author={A. Rousseau, P. Deléglise, and Y. Estève},
          journal={Proceedings of the Eighth International Conference on Language Resources and Evaluation (LREC'12)},
          year={May 2014},
          biburl={https://www.openslr.org/19/}
        }

        @article{
          title={TED-LIUM 3: twice as much data and corpus repartition for experiments on speaker adaptation},
          author={François Hernandez, Vincent Nguyen, Sahar Ghannay, Natalia Tomashenko, and Yannick Estève},
          journal={the 20th International Conference on Speech and Computer (SPECOM 2018)},
          year={September 2018},
          biburl={https://www.openslr.org/51/}
        }


.. include:: mindspore.dataset.api_list_audio.rst
