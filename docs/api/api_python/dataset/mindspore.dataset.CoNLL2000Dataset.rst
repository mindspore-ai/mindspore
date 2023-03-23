mindspore.dataset.CoNLL2000Dataset
==================================

.. py:class:: mindspore.dataset.CoNLL2000Dataset(dataset_dir, usage=None, num_samples=None, shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, num_parallel_workers=None, cache=None)

    CoNLL-2000（Conference on Computational Natural Language Learning）分块数据集。

    生成的数据集有三列 `[word, pos_tag, chunk_tag]` 。三列的数据类型均为string。

    参数：
        - **dataset_dir** (str) - 包含CoNLL2000分块数据集文件的根目录路径。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 'train'、'test' 或 'all'。
          'train' 将读取8936个训练样本，'test' 将读取2,012个测试样本中，'all' 将读取所有1,0948个样本。默认值：None，读取全部样本。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None，读取所有样本。
        - **shuffle** (Union[bool, Shuffle], 可选) - 每个epoch中数据混洗的模式。默认值：mindspore.dataset.Shuffle.GLOBAL。
          如果 `shuffle` 为False，则不混洗。如果 `shuffle` 为True，执行全局混洗。
          总共有三种混洗模式，通过枚举变量mindspore.dataset.Shuffle指定。

          - **Shuffle.GLOBAL**：混洗文件和样本，与设置为True效果相同。
          - **Shuffle.FILES**：仅混洗文件。

        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。指定此参数后， `num_samples` 表示每个分片的最大样本数。默认值：None。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。只有当指定了 `num_shards` 时才能指定此参数。默认值：None。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用全局默认线程数(8)，也可以通过 `mindspore.dataset.config.set_num_parallel_workers` 配置全局线程数。
        - **cache** (DatasetCache, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/cache.html>`_ 。默认值：None，不使用缓存。

    异常：
        - **RuntimeError** - `dataset_dir` 参数所指向的文件目录不存在或缺少数据集文件。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。

    **关于CoNLL2000数据集：**

    CoNLL2000分块数据集由华尔街日报语料库第15-20节的文本组成。
    文本使用IOB表示法进行分块，分块类型有NP, VP, PP, ADJP和ADVP。
    数据集由通过空格分隔的三列组成。第一列包含当前单词，第二列是由Brill标注器派生的词性标注，第三列是由华尔街语料库派生的分块标注。
    文本分块旨在将文本划分为单词的句法的相关组成部分。

    可以将数据集文件解压缩到以下结构，并通过MindSpore的API读取：

    .. code-block::

        .
        └── conll2000_dataset_dir
             ├── train.txt
             ├── test.txt
             └── readme.txt

    **引用：**

    .. code-block::

        @inproceedings{tksbuchholz2000conll,
        author     = {Tjong Kim Sang, Erik F. and Sabine Buchholz},
        title      = {Introduction to the CoNLL-2000 Shared Task: Chunking},
        editor     = {Claire Cardie and Walter Daelemans and Claire Nedellec and Tjong Kim Sang, Erik},
        booktitle  = {Proceedings of CoNLL-2000 and LLL-2000},
        publisher  = {Lisbon, Portugal},
        pages      = {127--132},
        year       = {2000}
        }


.. include:: mindspore.dataset.api_list_nlp.rst
