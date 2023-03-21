mindspore.dataset.WIDERFaceDataset
==================================

.. py:class:: mindspore.dataset.WIDERFaceDataset(dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None, decode=False, sampler=None, num_shards=None, shard_id=None, cache=None)

    WIDERFace数据集。

    当 `usage` 为 "train"、"valid" 或 "all" 时，生成的数据集有八列 `["image", "bbox", "blur", "expression", "illumination", "occlusion", "pose", "invalid"]` 。其中 `image` 列的数据类型为uint8，其他列均为uint32。
    当 `usage` 为 "test" 时，生成的数据集只有一列 `["image"]`，数据类型为uint8。

    参数：
        - **dataset_dir** (str) - 包含数据集文件的根目录路径。
        - **usage** (str, 可选) - 指定数据集的子集，可取值为 'train'、'test'、'valid' 或 'all'。
          取值为 'train' 时将会读取12,880个样本，取值为 'test' 时将会读取16,097个样本，取值为 'valid' 时将会读取3,226个样本，取值为 'all' 时将会读取全部类别样本。默认值：None，读取全部样本。
        - **num_samples** (int, 可选) - 指定从数据集中读取的样本数。默认值：None，读取全部样本。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作线程数。默认值：None，使用全局默认线程数(8)，也可以通过 `mindspore.dataset.config.set_num_parallel_workers` 配置全局线程数。
        - **shuffle** (bool, 可选) - 是否混洗数据集。默认值：None。下表中会展示不同参数配置的预期行为。
        - **decode** (bool, 可选) - 是否对读取的图片进行解码操作。默认值：False，不解码。
        - **sampler** (Sampler, 可选) - 指定从数据集中选取样本的采样器。默认值：None。下表中会展示不同配置的预期行为。
        - **num_shards** (int, 可选) - 指定分布式训练时将数据集进行划分的分片数。默认值：None。指定此参数后， `num_samples` 表示每个分片的最大样本数。
        - **shard_id** (int, 可选) - 指定分布式训练时使用的分片ID号。默认值：None。只有当指定了 `num_shards` 时才能指定此参数。
        - **cache** (DatasetCache, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/cache.html>`_ 。默认值：None，不使用缓存。

    异常：
        - **RuntimeError** - `dataset_dir` 不包含任何数据文件。
        - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 参数。
        - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 参数或同时指定了 `sampler` 和 `shard_id` 参数。
        - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError** - 如果 `shard_id` 取值不在[0, `num_shards` )范围。
        - **ValueError** - `usage` 不在['train', 'test', 'valid', 'all']中。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。
        - **ValueError** - `annotation_file` 不存在。
        - **ValueError** - `dataset_dir` 不存在。
    
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

    **关于WIDERFace数据集：**

    WIDER FACE数据集具有12,880个训练样本，16,097个测试样本，以及3,226个验证样本。此数据集是WIDER数据集的子集。其中图片已经预先进行了尺寸归一化和人像中心化处理。

    以下是原始的WIDERFace数据集结构。可以将数据集文件解压缩到此目录结构中，并由MindSpore的API读取。

    .. code-block::

        .
        └── wider_face_dir
             ├── WIDER_test
             │    └── images
             │         ├── 0--Parade
             │         │     ├── 0_Parade_marchingband_1_9.jpg
             │         │     ├── ...
             │         ├──1--Handshaking
             │         ├──...
             ├── WIDER_train
             │    └── images
             │         ├── 0--Parade
             │         │     ├── 0_Parade_marchingband_1_11.jpg
             │         │     ├── ...
             │         ├──1--Handshaking
             │         ├──...
             ├── WIDER_val
             │    └── images
             │         ├── 0--Parade
             │         │     ├── 0_Parade_marchingband_1_102.jpg
             │         │     ├── ...
             │         ├──1--Handshaking
             │         ├──...
             └── wider_face_split
                  ├── wider_face_test_filelist.txt
                  ├── wider_face_train_bbx_gt.txt
                  └── wider_face_val_bbx_gt.txt

    **引用：**

    .. code-block::

        @inproceedings{2016WIDER,
          title={WIDERFACE: A Detection Benchmark},
          author={Yang, S. and Luo, P. and Loy, C. C. and Tang, X.},
          booktitle={IEEE},
          pages={5525-5533},
          year={2016},
        }


.. include:: mindspore.dataset.api_list_vision.rst
