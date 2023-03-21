mindspore.dataset.FlickrDataset
================================

.. py:class:: mindspore.dataset.FlickrDataset(dataset_dir, annotation_file, num_samples=None, num_parallel_workers=None, shuffle=None, decode=None, sampler=None, num_shards=None, shard_id=None, cache=None)

    Flickr8k和Flickr30k数据集。

    生成的数据集有两列: `[image, annotation]`。 `image` 列的数据类型为uint8。 `annotation` 列是一个包含5个标注字符的张量，如["a", "b", "c", "d", "e"]。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **annotation_file** (str) - 数据集标注JSON文件的路径。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数，可以小于数据集总数。默认值：None，读取全部样本图片。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：使用全局默认线程数(8)，也可以通过 `mindspore.dataset.config.set_num_parallel_workers` 配置全局线程数。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值：None，表2中会展示不同参数配置的预期行为。
        - **decode** (bool, 可选) - 是否对读取的图片进行解码操作。默认值：None，不解码。
        - **sampler** (Sampler, 可选) - 指定从数据集中选取样本的采样器。默认值：None，表2中会展示不同配置的预期行为。
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
        - **ValueError** - `annotation_file` 参数对应的文件不存在。
        - **ValueError** - `dataset_dir` 参数路径不存在。
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

    **关于Flickr8k数据集：**

    Flickr8k数据集由8092张彩色图像组成。Flickr8k.token.txt中有40460个标注，每张图像有5个标注。

    可以将数据集文件解压缩到以下目录结构中，并由MindSpore的API读取。

    .. code-block::

        .
        └── Flickr8k
             ├── Flickr8k_Dataset
             │    ├── 1000268201_693b08cb0e.jpg
             │    ├── 1001773457_577c3a7d70.jpg
             │    ├── ...
             └── Flickr8k.token.txt

    **引用：**

    .. code-block::

        @article{DBLP:journals/jair/HodoshYH13,
        author    = {Micah Hodosh and Peter Young and Julia Hockenmaier},
        title     = {Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics},
        journal   = {J. Artif. Intell. Res.},
        volume    = {47},
        pages     = {853--899},
        year      = {2013},
        url       = {https://doi.org/10.1613/jair.3994},
        doi       = {10.1613/jair.3994},
        timestamp = {Mon, 21 Jan 2019 15:01:17 +0100},
        biburl    = {https://dblp.org/rec/journals/jair/HodoshYH13.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
        }

    **关于Flickr30k数据集：**

    Flickr30k数据集由31783张彩色图像组成。results_20130124.token中有158915个标注，每个图像有5个标注。

    可以将数据集文件解压缩到以下目录结构中，并由MindSpore的API读取。

    .. code-block::

        .
        └── Flickr30k
             ├── flickr30k-images
             │    ├── 1000092795.jpg
             │    ├── 10002456.jpg
             │    ├── ...
             └── results_20130124.token

    **引用：**

    .. code-block::

        @article{DBLP:journals/tacl/YoungLHH14,
        author    = {Peter Young and Alice Lai and Micah Hodosh and Julia Hockenmaier},
        title     = {From image descriptions to visual denotations: New similarity metrics
                     for semantic inference over event descriptions},
        journal   = {Trans. Assoc. Comput. Linguistics},
        volume    = {2},
        pages     = {67--78},
        year      = {2014},
        url       = {https://tacl2013.cs.columbia.edu/ojs/index.php/tacl/article/view/229},
        timestamp = {Wed, 17 Feb 2021 21:55:25 +0100},
        biburl    = {https://dblp.org/rec/journals/tacl/YoungLHH14.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
        }


.. include:: mindspore.dataset.api_list_vision.rst
