mindspore.dataset.dataloader
============================

此模块提供用于加载数据集的迭代器。支持加载 Map Style 和 Iterable Style 数据集，并提供多进程并发加载。

数据加载器
------------

.. mscnautosummary::
    :toctree: dataset_dataloader
    :nosignatures:
    :template: classtemplate_inherited.rst

    mindspore.dataset.dataloader.DataLoader

数据集
---------

.. mscnautosummary::
    :toctree: dataset_dataloader
    :nosignatures:
    :template: classtemplate_inherited.rst

    mindspore.dataset.dataloader.Dataset
    mindspore.dataset.dataloader.IterableDataset
    mindspore.dataset.dataloader.TensorDataset

采样器
---------

.. mscnautosummary::
    :toctree: dataset_dataloader
    :nosignatures:
    :template: classtemplate_inherited.rst

    mindspore.dataset.dataloader.Sampler
    mindspore.dataset.dataloader.SequentialSampler
    mindspore.dataset.dataloader.RandomSampler
    mindspore.dataset.dataloader.BatchSampler
    mindspore.dataset.dataloader.DistributedSampler

整理函数
-----------

.. mscnautosummary::
    :toctree: dataset_dataloader
    :nosignatures:
    :template: classtemplate_inherited.rst

    mindspore.dataset.dataloader.default_collate
    mindspore.dataset.dataloader.default_convert
    mindspore.dataset.dataloader._utils.collate.collate

工具
-------

.. mscnautosummary::
    :toctree: dataset_dataloader
    :nosignatures:
    :template: classtemplate_inherited.rst

    mindspore.dataset.dataloader.get_worker_info
