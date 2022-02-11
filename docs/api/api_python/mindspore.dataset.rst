mindspore.dataset
=================

该模块提供了加载和处理各种通用数据集的API，如MNIST、CIFAR-10、CIFAR-100、VOC、COCO、ImageNet、CelebA、CLUE等，
也支持加载业界标准格式的数据集，包括MindRecord、TFRecord、Manifest等。此外，用户还可以使用此模块定义和加载自己的数据集。

该模块还提供了在加载时进行数据采样的API，如SequentialSample、RandomSampler、DistributedSampler等。

大多数数据集可以通过指定参数 `cache` 启用缓存服务，以提升整体数据处理效率。
请注意Windows平台上还不支持缓存服务，因此在Windows上加载和处理数据时，请勿使用。更多介绍和限制，
请参考 `Single-Node Tensor Cache <https://www.mindspore.cn/docs/programming_guide/zh-CN/master/cache.html>`_。

在API示例中，常用的模块导入方法如下：

.. code-block::

    import mindspore.dataset as ds
    from mindspore.dataset.transforms import c_transforms

常用数据集术语说明如下：

- Dataset，所有数据集的基类，提供了数据处理方法来帮助预处理数据。
- SourceDataset，一个抽象类，表示数据集管道的来源，从文件和数据库等数据源生成数据。
- MappableDataset，一个抽象类，表示支持随机访问的源数据集。
- Iterator，用于枚举元素的数据集迭代器的基类。

视觉
-----

.. cnmsautosummary::
    :toctree: dataset

    mindspore.dataset.CelebADataset
    mindspore.dataset.CocoDataset
    mindspore.dataset.ImageFolderDataset
    mindspore.dataset.MnistDataset
    mindspore.dataset.VOCDataset
    mindspore.dataset.Cifar100Dataset
    mindspore.dataset.Cifar10Dataset

文本
----

.. cnmsautosummary::
    :toctree: dataset

    mindspore.dataset.CLUEDataset

图
---

.. cnmsautosummary::
    :toctree: dataset
    
    mindspore.dataset.GraphData


用户自定义
----------

.. cnmsautosummary::
    :toctree: dataset

    mindspore.dataset.GeneratorDataset
    mindspore.dataset.NumpySlicesDataset
    mindspore.dataset.PaddedDataset

采样器
-------

.. cnmsautosummary::
    :toctree: dataset

    mindspore.dataset.DistributedSampler
    mindspore.dataset.PKSampler
    mindspore.dataset.RandomSampler
    mindspore.dataset.SequentialSampler
    mindspore.dataset.SubsetRandomSampler
    mindspore.dataset.SubsetSampler
    mindspore.dataset.WeightedRandomSampler

标准格式
--------

.. cnmsautosummary::
    :toctree: dataset

    mindspore.dataset.CSVDataset
    mindspore.dataset.ManifestDataset
    mindspore.dataset.MindDataset
    mindspore.dataset.TFRecordDataset
    mindspore.dataset.TextFileDataset

其他
-----

.. cnmsautosummary::
    :toctree: dataset

    mindspore.dataset.DSCallback
    mindspore.dataset.DatasetCache
    mindspore.dataset.Schema
    mindspore.dataset.WaitedDSCallback
    mindspore.dataset.compare
    mindspore.dataset.deserialize
    mindspore.dataset.serialize
    mindspore.dataset.show
    mindspore.dataset.utils.imshow_det_bbox
    mindspore.dataset.zip
