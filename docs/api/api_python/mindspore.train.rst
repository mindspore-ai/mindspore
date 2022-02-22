mindspore.train
===============

mindspore.train.summary
------------------------

使用SummaryRecord将需要的数据存储为summary文件和lineage文件，使用方法包括自定义回调函数和自定义训练循环。保存的summary文件使用MindInsight进行可视化分析。

.. include:: train/mindspore.train.summary.SummaryRecord.rst

mindspore.train.callback
------------------------

.. include:: train/mindspore.train.callback.Callback.rst

.. include:: train/mindspore.train.callback.LossMonitor.rst

.. include:: train/mindspore.train.callback.TimeMonitor.rst

.. include:: train/mindspore.train.callback.ModelCheckpoint.rst

.. include:: train/mindspore.train.callback.SummaryCollector.rst

.. include:: train/mindspore.train.callback.CheckpointConfig.rst

.. include:: train/mindspore.train.callback.RunContext.rst

.. include:: train/mindspore.train.callback.LearningRateScheduler.rst

.. include:: train/mindspore.train.callback.SummaryLandscape.rst

.. automodule:: mindspore.train.callback
    :exclude-members: FederatedLearningManager
    :members:

mindspore.train.train_thor
--------------------------

转换为二阶相关的类和函数。

.. include:: train/mindspore.train.train_thor.ConvertModelUtils.rst

.. include:: train/mindspore.train.train_thor.ConvertNetUtils.rst
