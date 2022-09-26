mindspore.train
===============

模型
-----

.. mscnautosummary::
    :toctree: train

    mindspore.train.Model

回调函数
---------

.. mscnautosummary::
    :toctree: train

    mindspore.train.Callback
    mindspore.train.CheckpointConfig
    mindspore.train.EarlyStopping
    mindspore.train.History
    mindspore.train.LambdaCallback
    mindspore.train.LearningRateScheduler
    mindspore.train.LossMonitor
    mindspore.train.ModelCheckpoint
    mindspore.train.OnRequestExit
    mindspore.train.ReduceLROnPlateau
    mindspore.train.RunContext
    mindspore.train.TimeMonitor

评价指标
--------

.. mscnplatformautosummary::
    :toctree: train
    :nosignatures:
    :template: classtemplate.rst

    mindspore.train.Accuracy
    mindspore.train.BleuScore
    mindspore.train.ConfusionMatrix
    mindspore.train.ConfusionMatrixMetric
    mindspore.train.CosineSimilarity
    mindspore.train.Dice
    mindspore.train.F1
    mindspore.train.Fbeta
    mindspore.train.HausdorffDistance
    mindspore.train.Loss
    mindspore.train.MAE
    mindspore.train.MeanSurfaceDistance
    mindspore.train.Metric
    mindspore.train.MSE
    mindspore.train.OcclusionSensitivity
    mindspore.train.Perplexity
    mindspore.train.Precision
    mindspore.train.Recall
    mindspore.train.ROC
    mindspore.train.RootMeanSquareDistance
    mindspore.train.Top1CategoricalAccuracy
    mindspore.train.Top5CategoricalAccuracy
    mindspore.train.TopKCategoricalAccuracy

工具
----

.. mscnplatformautosummary::
    :toctree: train
    :nosignatures:
    :template: classtemplate.rst

    mindspore.train.auc
    mindspore.train.get_metric_fn
    mindspore.train.names
    mindspore.train.rearrange_inputs
