mindspore.train
===============

mindspore.train.summary
------------------------

使用SummaryRecord将需要的数据存储为summary文件和lineage文件，使用方法包括自定义回调函数和自定义训练循环。保存的summary文件使用MindInsight进行可视化分析。

.. include:: mindspore.train/mindspore.train.summary.SummaryRecord.rst

mindspore.train.callback
------------------------

.. include:: mindspore.train/mindspore.train.callback.Callback.rst

.. include:: mindspore.train/mindspore.train.callback.LossMonitor.rst

.. include:: mindspore.train/mindspore.train.callback.TimeMonitor.rst

.. include:: mindspore.train/mindspore.train.callback.ModelCheckpoint.rst

.. include:: mindspore.train/mindspore.train.SummaryCollector.rst

.. include:: mindspore.train/mindspore.train.callback.CheckpointConfig.rst

.. include:: mindspore.train/mindspore.train.callback.RunContext.rst

.. include:: mindspore.train/mindspore.train.callback.LearningRateScheduler.rst
