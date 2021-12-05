mindspore.dataset.CLUEDataset
=============================

.. py:class:: mindspore.dataset.CLUEDataset(dataset_files, task='AFQMC', usage='train', num_samples=None, num_parallel_workers=None, shuffle=<Shuffle.GLOBAL: 'global'>, num_shards=None, shard_id=None, cache=None)

    读取和解析CLUE数据集的源数据集文件。目前支持的CLUE分类任务包括： `AFQMC` 、 `Tnews` 、 `IFLYTEK` 、 `CMNLI` 、 `WSC` 和 `CSL` 。

    根据给定的 `task` 配置，数据集会生成不同的输出列：

    - task = `AFQMC`
        - usage = `train`，输出列: `[sentence1, dtype=string]`, `[sentence2, dtype=string]`, `[label, dtype=string]`.
        - usage = `test`，输出列: `[id, dtype=uint8]`, `[sentence1, dtype=string]`, `[sentence2, dtype=string]`.
        - usage = `eval`，输出列: `[sentence1, dtype=string]`, `[sentence2, dtype=string]`, `[label, dtype=string]`.

    - task = `TNEWS`
        - usage = `train`，输出列: `[label, dtype=string]`, `[label_des, dtype=string]`, `[sentence, dtype=string]`, `[keywords, dtype=string]`.
        - usage = `test`，输出列: `[label, dtype=string]`, `[label_des, dtype=string]`, `[sentence, dtype=string]`, `[keywords, dtype=string]`.
        - usage = `eval`，输出列: `[label, dtype=string]`, `[label_des, dtype=string]`, `[sentence, dtype=string]`, `[keywords, dtype=string]`.

    - task = `IFLYTEK`
        - usage = `train`，输出列: `[label, dtype=string]`, `[label_des, dtype=string]`, `[sentence, dtype=string]`.
        - usage = `test`，输出列: `[id, dtype=string]`, `[sentence, dtype=string]`.
        - usage = `eval`，输出列: `[label, dtype=string]`, `[label_des, dtype=string]`, `[sentence, dtype=string]`.

    - task = `CMNLI`
        - usage = `train`，输出列: `[sentence1, dtype=string]`, `[sentence2, dtype=string]`, `[label, dtype=string]`.
        - usage = `test`，输出列: `[id, dtype=uint8]`, `[sentence1, dtype=string]`, `[sentence2, dtype=string]`.
        - usage = `eval`，输出列: `[sentence1, dtype=string]`, `[sentence2, dtype=string]`, `[label, dtype=string]`.

    - task = `WSC`
        - usage = `train`，输出列: `[span1_index, dtype=uint8]`, `[span2_index, dtype=uint8]`, `[span1_text, dtype=string]`, `[span2_text, dtype=string]`, `[idx, dtype=uint8]`, `[text, dtype=string]`, `[label, dtype=string]`.
        - usage = `test`，输出列: `[span1_index, dtype=uint8]`, `[span2_index, dtype=uint8]`, `[span1_text, dtype=string]`, `[span2_text, dtype=string]`, `[idx, dtype=uint8]`, `[text, dtype=string]`.
        - usage = `eval`，输出列: `[span1_index, dtype=uint8]`, `[span2_index, dtype=uint8]`, `[span1_text, dtype=string]`, `[span2_text, dtype=string]`, `[idx, dtype=uint8]`, `[text, dtype=string]`, `[label, dtype=string]`.

    - task = `CSL`
        - usage = `train`，输出列: `[id, dtype=uint8]`, `[abst, dtype=string]`, `[keyword, dtype=string]`, `[label, dtype=string]`.
        - usage = `test`，输出列: `[id, dtype=uint8]`, `[abst, dtype=string]`, `[keyword, dtype=string]`.
        - usage = `eval`，输出列: `[id, dtype=uint8]`, `[abst, dtype=string]`, `[keyword, dtype=string]`, `[label, dtype=string]`.

    **参数：**

    - **dataset_files** (Union[str, list[str]]) - 数据集文件路径，支持单文件路径字符串、多文件路径字符串列表或可被glob库模式匹配的字符串，文件列表将在内部进行字典排序。
    - **task** (str, 可选) - 任务类型，可取值为 `AFQMC` 、`Tnews`、`IFLYTEK`、`CMNLI`、`WSC` 或 `CSL` （默认为： `AFQMC` ）。
    - **usage** (str, 可选) - 指定数据集的子集，可取值为 `train`、`test` 或 `eval` （默认为： `train` ）。
    - **num_samples** (int, 可选) - 指定从数据集中读取的样本数（默认为None，即读取所有图像样本）。
    - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数(默认值None，即使用mindspore.dataset.config中配置的线程数）。
    - **shuffle** (Union[bool, Shuffle level], 可选) - 每个epoch中数据混洗的模式（默认为为mindspore.dataset.Shuffle.GLOBAL）。如果为False，则不混洗；如果为True，等同于将 `shuffle` 设置为mindspore.dataset.Shuffle.GLOBAL。另外也可以传入枚举变量设置shuffle级别：

      - **Shuffle.GLOBAL**：混洗文件和样本。
      - **Shuffle.FILES**：仅混洗文件。

    - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数（默认值None）。指定此参数后, `num_samples` 表示每个分片的最大样本数。
    - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号（默认值None）。只有当指定了 `num_shards` 时才能指定此参数。
    - **cache** (DatasetCache, 可选) - 数据缓存客户端实例，用于加快数据集处理速度（默认为None，不使用缓存）。

    **异常：**

    - **RuntimeError** - `dataset_files` 所指的文件无效或不存在。
    - **RuntimeError** - `num_parallel_workers` 超过系统最大线程数。
    - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
    - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。

    **样例：**

    >>> clue_dataset_dir = ["/path/to/clue_dataset_file"] # 包含一个或多个CLUE数据集文件
    >>> dataset = ds.CLUEDataset(dataset_files=clue_dataset_dir, task='AFQMC', usage='train')

    **关于CLUE数据集：**

    CLUE，又名中文语言理解测评基准，包含许多有代表性的数据集，涵盖单句分类、句对分类和机器阅读理解等任务。

    您可以将数据集解压成如下的文件结构，并通过MindSpore的API进行读取，以 `afqmc` 数据集为例：

    .. code-block::

        .
        └── afqmc_public
             ├── train.json
             ├── test.json
             └── dev.json

    **引用：**

    .. code-block::

        @article{CLUEbenchmark,
        title   = {CLUE: A Chinese Language Understanding Evaluation Benchmark},
        author  = {Liang Xu, Xuanwei Zhang, Lu Li, Hai Hu, Chenjie Cao, Weitang Liu, Junyi Li, Yudong Li,
                Kai Sun, Yechen Xu, Yiming Cui, Cong Yu, Qianqian Dong, Yin Tian, Dian Yu, Bo Shi, Jun Zeng,
                Rongzhao Wang, Weijian Xie, Yanting Li, Yina Patterson, Zuoyu Tian, Yiwen Zhang, He Zhou,
                Shaoweihua Liu, Qipeng Zhao, Cong Yue, Xinrui Zhang, Zhengliang Yang, Zhenzhong Lan},
        journal = {arXiv preprint arXiv:2004.05986},
        year    = {2020},
        howpublished = {https://github.com/CLUEbenchmark/CLUE}
        }

    .. include:: mindspore.dataset.Dataset.rst

    .. include:: mindspore.dataset.Dataset.zip.rst