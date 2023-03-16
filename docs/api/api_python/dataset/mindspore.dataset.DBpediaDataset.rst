mindspore.dataset.DBpediaDataset
================================

.. py:class:: mindspore.dataset.DBpediaDataset(dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, cache=None)

    读取和解析DBpedia数据集的源数据集。

    生成的数据集有三列 `[class, title, content]` ，三列的数据类型均为string。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 'train'，'test' 或 'all'。
          'train' 将读取560,000个训练样本，'test' 将读取70,000个测试样本中，'all' 将读取所有630,000个样本。默认值：None，读取全部样本。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None，读取所有样本。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用 `mindspore.dataset.config` 中配置的线程数。
        - **shuffle** (Union[bool, Shuffle], 可选) - 每个epoch中数据混洗的模式，支持传入bool类型与枚举类型进行指定。默认值：`Shuffle.GLOBAL` 。
          如果 `shuffle` 为False，则不混洗，如果 `shuffle` 为True，等同于将 `shuffle` 设置为mindspore.dataset.Shuffle.GLOBAL。
          通过传入枚举变量设置数据混洗的模式：

          - **Shuffle.GLOBAL**：混洗文件和样本。
          - **Shuffle.FILES**：仅混洗文件。

        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值：None。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值：None。只有当指定了 `num_shards` 时才能指定此参数。
        - **cache** (DatasetCache, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/cache.html>`_ 。默认值：None，不使用缓存。

    异常：
        - **RuntimeError** - `dataset_dir` 参数所指向的文件目录不存在或缺少数据集文件。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。
        - **ValueError** - 如果 `shard_id` 取值不在[0, `num_shards` )范围。

    **关于DBpedia数据集：**

    DBpedia数据集包括14个类，超过63万个文本样本，train.csv中有56万样本，test.csv中有7万测试样本。
    14个不同的类别分别是：公司、教育学院、艺术家、运动员、文员，交通，建筑，自然场所，村庄，动物，植物，专辑，电影，书面工作。

    以下是原始DBpedia数据集结构。
    可以将数据集文件解压缩到此目录结构中，并通过Mindspore的API读取。

    .. code-block::

        .
        └── dbpedia_dataset_dir
            ├── train.csv
            ├── test.csv
            ├── classes.txt
            └── readme.txt

    **引用：**

    .. code-block::

        @article{DBpedia,
        title   = {DBPedia Ontology Classification Dataset},
        author  = {Jens Lehmann, Robert Isele, Max Jakob, Anja Jentzsch, Dimitris Kontokostas,
                Pablo N. Mendes, Sebastian Hellmann, Mohamed Morsey, Patrick van Kleef,
                    Sören Auer, Christian Bizer},
        year    = {2015},
        howpublished = {http://dbpedia.org}
        }


.. include:: mindspore.dataset.api_list_nlp.rst
