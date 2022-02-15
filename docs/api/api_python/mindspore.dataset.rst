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

.. mscnautosummary::
    :toctree: dataset
    :nosignatures:
    :template: classtemplate_inherited.rst

    mindspore.dataset.Caltech101Dataset
    mindspore.dataset.Caltech256Dataset
    mindspore.dataset.CelebADataset
    mindspore.dataset.Cifar10Dataset
    mindspore.dataset.Cifar100Dataset
    mindspore.dataset.CityscapesDataset
    mindspore.dataset.CocoDataset
    mindspore.dataset.DIV2KDataset
    mindspore.dataset.EMnistDataset
    mindspore.dataset.FakeImageDataset
    mindspore.dataset.FashionMnistDataset
    mindspore.dataset.FlickrDataset
    mindspore.dataset.Flowers102Dataset
    mindspore.dataset.ImageFolderDataset
    mindspore.dataset.KMnistDataset
    mindspore.dataset.ManifestDataset
    mindspore.dataset.MnistDataset
    mindspore.dataset.PhotoTourDataset
    mindspore.dataset.Places365Dataset
    mindspore.dataset.QMnistDataset
    mindspore.dataset.SBDataset
    mindspore.dataset.SBUDataset
    mindspore.dataset.SemeionDataset
    mindspore.dataset.STL10Dataset
    mindspore.dataset.SVHNDataset
    mindspore.dataset.USPSDataset
    mindspore.dataset.VOCDataset
    mindspore.dataset.WIDERFaceDataset

文本
----

.. mscnautosummary::
    :toctree: dataset
    :nosignatures:
    :template: classtemplate_inherited.rst

    mindspore.dataset.AGNewsDataset
    mindspore.dataset.AmazonReviewDataset
    mindspore.dataset.CLUEDataset
    mindspore.dataset.CoNLL2000Dataset
    mindspore.dataset.CSVDataset
    mindspore.dataset.DBpediaDataset
    mindspore.dataset.EnWik9Dataset
    mindspore.dataset.IMDBDataset
    mindspore.dataset.IWSLT2016Dataset
    mindspore.dataset.IWSLT2017Dataset
    mindspore.dataset.PennTreebankDataset
    mindspore.dataset.SogouNewsDataset
    mindspore.dataset.TextFileDataset
    mindspore.dataset.UDPOSDataset
    mindspore.dataset.WikiTextDataset
    mindspore.dataset.YahooAnswersDataset
    mindspore.dataset.YelpReviewDataset

音频
------

.. mscnautosummary::
    :toctree: dataset
    :nosignatures:
    :template: classtemplate_inherited.rst

    mindspore.dataset.LJSpeechDataset
    mindspore.dataset.SpeechCommandsDataset
    mindspore.dataset.TedliumDataset
    mindspore.dataset.YesNoDataset

标准格式
--------

.. mscnautosummary::
    :toctree: dataset
    :nosignatures:
    :template: classtemplate_inherited.rst

    mindspore.dataset.CSVDataset
    mindspore.dataset.MindDataset
    mindspore.dataset.TFRecordDataset

用户自定义
----------

.. mscnautosummary::
    :toctree: dataset
    :nosignatures:
    :template: classtemplate_inherited.rst

    mindspore.dataset.GeneratorDataset
    mindspore.dataset.NumpySlicesDataset
    mindspore.dataset.PaddedDataset
    mindspore.dataset.RandomDataset

图
---

.. mscnautosummary::
    :toctree: dataset
    
    mindspore.dataset.GraphData


采样器
-------

.. mscnautosummary::
    :toctree: dataset

    mindspore.dataset.DistributedSampler
    mindspore.dataset.PKSampler
    mindspore.dataset.RandomSampler
    mindspore.dataset.SequentialSampler
    mindspore.dataset.SubsetRandomSampler
    mindspore.dataset.SubsetSampler
    mindspore.dataset.WeightedRandomSampler


其他
-----

.. mscnautosummary::
    :toctree: dataset
    :nosignatures:
    :template: classtemplate_inherited.rst

    mindspore.dataset.BatchInfo
    mindspore.dataset.DatasetCache
    mindspore.dataset.DSCallback
    mindspore.dataset.SamplingStrategy
    mindspore.dataset.Schema
    mindspore.dataset.Shuffle
    mindspore.dataset.WaitedDSCallback
    mindspore.dataset.OutputFormat
    mindspore.dataset.compare
    mindspore.dataset.deserialize
    mindspore.dataset.serialize
    mindspore.dataset.show
    mindspore.dataset.utils.imshow_det_bbox
    mindspore.dataset.zip
