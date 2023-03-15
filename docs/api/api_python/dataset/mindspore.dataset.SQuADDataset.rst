mindspore.dataset.SQuADDataset
==============================

.. py:class:: mindspore.dataset.SQuADDataset(dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None, num_shards=None, shard_id=None, cache=None)

    读取和解析SQuAD 1.1或SQuAD 2.0数据集的源文件构建数据集。

    不同版本和子集生成的数据集具有相同的列: `[context, question, text, answer_start]`。
    `context` 列的数据类型为string。
    `question` 列的数据类型为string。
    `text` 列为上下文中的回答，数据类型为string。
    `answer_start` 列为上下文中回答的起始索引，数据类型为uint32。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 'train'、'dev' 或 'all'。默认值：None，读取全部样本。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None，读取全部样本。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用mindspore.dataset.config中配置的线程数。
        - **shuffle** (Union[bool, Shuffle], 可选) - 是否混洗数据集。默认值：None，表示 Shuffle.GLOBAL 。
          如果输入False，将不进行混洗。
          如果输入True，效果与设置 Shuffle.GLOBAL 相同。
          如果输入Shuffle枚举值，效果如下表所示：

          - **Shuffle.GLOBAL**：混洗文件和文件中的数据。
          - **Shuffle.FILES**：仅混洗文件。

        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值：None。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值：None。只有当指定了 `num_shards` 时才能指定此参数。
        - **cache** (DatasetCache, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/cache.html>`_ 。默认值：None，不使用缓存。

    异常：
        - **RuntimeError** - `dataset_dir` 路径下不包含数据文件。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - `shard_id` 参数错误，小于0或者大于等于 `num_shards` 。

    **关于SQuAD数据集：**

    SQuAD（Stanford Question Answering Dataset）是一个阅读理解数据集，由众人对一组维基百科文章提出的问题组成，
    每个问题的答案都是相应阅读段落中的一段文字或范围，否则问题可能无法回答。

    SQuAD 1.1，即SQuAD数据集的前一个版本，包含500多篇文章的100,000多个问题-答案对。SQuAD 2.0除包含SQuAD 1.1中的
    100,000个问题外，还补充了超过50,000个由贡献者编写的不可回答的对抗性问题，它们看起来与可回答的问题类似。为了
    在SQuAD 2.0中取得好成绩，系统不仅要尽量回答可回答的问题，而且要能够在段落中不存在答案时放弃回答。

    您可以将数据集解压并构建成以下目录结构，并通过MindSpore的API进行读取。

    SQuAD 1.1:

    .. code-block::

        .
        └── SQuAD1
             ├── train-v1.1.json
             └── dev-v1.1.json

    .. code-block::

        .
        └── SQuAD2
             ├── train-v2.0.json
             └── dev-v2.0.json

    **引用：**

    .. code-block::

        @misc{rajpurkar2016squad,
            title         = {SQuAD: 100,000+ Questions for Machine Comprehension of Text},
            author        = {Pranav Rajpurkar and Jian Zhang and Konstantin Lopyrev and Percy Liang},
            year          = {2016},
            eprint        = {1606.05250},
            archivePrefix = {arXiv},
            primaryClass  = {cs.CL}
        }

        @misc{rajpurkar2018know,
            title         = {Know What You Don't Know: Unanswerable Questions for SQuAD},
            author        = {Pranav Rajpurkar and Robin Jia and Percy Liang},
            year          = {2018},
            eprint        = {1806.03822},
            archivePrefix = {arXiv},
            primaryClass  = {cs.CL}
        }


.. include:: mindspore.dataset.api_list_nlp.rst
