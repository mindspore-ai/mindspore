mindspore.dataset.YelpReviewDataset
===================================

.. py:class:: mindspore.dataset.YelpReviewDataset(dataset_dir, usage=None, num_samples=None, shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, num_parallel_workers=None, cache=None)

    读取和解析Yelp Review Full和Yelp Review Polarity数据集的源数据集。

    生成的数据集有两列 `[label, text]`，两列的数据类型均为string。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 'train'、'test' 或 'all'。默认值：None，读取全部样本。
          对于Polarity数据集，'train' 将读取560,000个训练样本，'test' 将读取38,000个测试样本，'all' 将读取所有598,000个样本。
          对于Full数据集，'train' 将读取650,000个训练样本，'test' 将读取50,000个测试样本，'all' 将读取所有700,000个样本。默认值：None，读取所有样本。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None，读取全部样本。
        - **shuffle** (Union[bool, Shuffle], 可选) - 每个epoch中数据混洗的模式，支持传入bool类型与枚举类型进行指定。默认值：`Shuffle.GLOBAL` 。
          如果 `shuffle` 为False，则不混洗，如果 `shuffle` 为True，等同于将 `shuffle` 设置为mindspore.dataset.Shuffle.GLOBAL。
          通过传入枚举变量设置数据混洗的模式：

          - **Shuffle.GLOBAL**：混洗文件和样本。
          - **Shuffle.FILES**：仅混洗文件。

        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值：None。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值：None。只有当指定了 `num_shards` 时才能指定此参数。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用 `mindspore.dataset.config` 中配置的线程数。
        - **cache** (DatasetCache, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/cache.html>`_ 。默认值：None，不使用缓存。

    异常：
        - **RuntimeError** - `dataset_dir` 参数所指向的文件目录不存在或缺少数据集文件。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。

    **关于YelpReview数据集：**

    Yelp Review Full数据集包括来自Yelp的评论数据。这些数据时从2015年的Yelp数据集挑战赛数据中提取的，主要用于文本分类。

    Yelp Review Polarity数据集在Full数据集的基础上，对产品评分进行了分级，评论分数1和2视为负面评论，4和5视为正面评论。

    Yelp Reviews Polarity和Yelp Reviews Full datasets具有相同的目录结构。
    可以将数据集文件解压缩到以下结构，并通过MindSpore的API读取：

    .. code-block::

        .
        └── yelp_review_dir
             ├── train.csv
             ├── test.csv
             └── readme.txt

    **引用：**

    .. code-block::

        @article{zhangCharacterlevelConvolutionalNetworks2015,
          archivePrefix = {arXiv},
          eprinttype = {arxiv},
          eprint = {1509.01626},
          primaryClass = {cs},
          title = {Character-Level {{Convolutional Networks}} for {{Text Classification}}},
          abstract = {This article offers an empirical exploration on the use of character-level convolutional networks
                      (ConvNets) for text classification. We constructed several large-scale datasets to show that
                      character-level convolutional networks could achieve state-of-the-art or competitive results.
                      Comparisons are offered against traditional models such as bag of words, n-grams and their TFIDF
                      variants, and deep learning models such as word-based ConvNets and recurrent neural networks.},
          journal = {arXiv:1509.01626 [cs]},
          author = {Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
          month = sep,
          year = {2015},
        }
    
    .. code-block::

        @article{zhangCharacterlevelConvolutionalNetworks2015,
          archivePrefix = {arXiv},
          eprinttype = {arxiv},
          eprint = {1509.01626},
          primaryClass = {cs},
          title = {Character-Level {{Convolutional Networks}} for {{Text Classification}}},
          abstract = {This article offers an empirical exploration on the use of character-level convolutional networks
                      (ConvNets) for text classification. We constructed several large-scale datasets to show that
                      character-level convolutional networks could achieve state-of-the-art or competitive results.
                      Comparisons are offered against traditional models such as bag of words, n-grams and their TFIDF
                      variants, and deep learning models such as word-based ConvNets and recurrent neural networks.},
          journal = {arXiv:1509.01626 [cs]},
          author = {Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
          month = sep,
          year = {2015},
        }


.. include:: mindspore.dataset.api_list_nlp.rst
