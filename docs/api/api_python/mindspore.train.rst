mindspore.train
===============

mindspore.train.summary
------------------------

用户可以通过SummaryRecord来自定义回调函数或者在自定义训练循环中将需要的数据存储为summary文件和lineage文件，然后使用MindInsight进行可视化分析。

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
