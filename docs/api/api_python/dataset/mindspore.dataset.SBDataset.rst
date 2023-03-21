mindspore.dataset.SBDataset
===========================

.. py:class:: mindspore.dataset.SBDataset(dataset_dir, task='Boundaries', usage='all', num_samples=None, num_parallel_workers=1, shuffle=None, decode=None, sampler=None, num_shards=None, shard_id=None)

    SB（Semantic Boundaries）数据集。

    通过配置 `task` 参数，生成的数据集具有不同的输出列：

    - `task` = 'Boundaries'，有两个输出列： `image` 列的数据类型为uint8，`label` 列包含1个的数据类型为uint8的图像。
    - `task` = 'Segmentation'，有两个输出列： `image` 列的数据类型为uint8。 `label` 列包含20个的数据类型为uint8的图像。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录的路径。
        - **task** (str, 可选) - 指定读取SB数据集的任务类型，支持 'Boundaries' 和 'Segmentation'。默认值：'Boundaries'。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 'train'、'val'、'train_noval' 和 'all'。默认值：'train'。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None，所有图像样本。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作进程数。默认值：1。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值：None。下表中会展示不同参数配置的预期行为。
        - **decode** (bool, 可选) - 是否对读取的图片进行解码操作。默认值：False，不解码。
        - **sampler** (Sampler, 可选) - 指定从数据集中选取样本的采样器。默认值：None。下表中会展示不同配置的预期行为。
        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值：None。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值：None。只有当指定了 `num_shards` 时才能指定此参数。

    异常：
        - **RuntimeError** - `dataset_dir` 路径下不包含任何数据文件。
        - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 参数。
        - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 参数或同时指定了 `sampler` 和 `shard_id` 参数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - `dataset_dir` 不存在。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。
        - **ValueError** - `task` 不是['Boundaries', 'Segmentation']中的任何一个。
        - **ValueError** - `usage` 不是['train', 'val', 'train_noval', 'all']中的任何一个。
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

    **关于Semantic Boundaries数据集：**

    Semantic Boundaries（语义边界）数据集由11355张彩色图像组成。
    train.txt中有8498个图像，val.txt中有2857个图像，train_noval.txt中有5623个图像。
    目录cls中包含类别的分割和边界标注，目录inst中包含实例级的分割和边界标注。

    可以将数据集文件解压缩为以下结构，并通过MindSpore的API读取：

    .. code-block::

         .
         └── benchmark_RELEASE
              ├── dataset
              ├── img
              │    ├── 2008_000002.jpg
              │    ├── 2008_000003.jpg
              │    ├── ...
              ├── cls
              │    ├── 2008_000002.mat
              │    ├── 2008_000003.mat
              │    ├── ...
              ├── inst
              │    ├── 2008_000002.mat
              │    ├── 2008_000003.mat
              │    ├── ...
              ├── train.txt
              └── val.txt

    **引用：**

    .. code-block::

        @InProceedings{BharathICCV2011,
            author       = "Bharath Hariharan and Pablo Arbelaez and Lubomir Bourdev and
                            Subhransu Maji and Jitendra Malik",
            title        = "Semantic Contours from Inverse Detectors",
            booktitle    = "International Conference on Computer Vision (ICCV)",
            year         = "2011",
        }


.. include:: mindspore.dataset.api_list_vision.rst
