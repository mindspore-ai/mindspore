mindspore.nn
=============

神经网络Cell。

用于构建神经网络中的预定义构建块或计算单元。

基本构成单元
------------

.. mscnplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.Cell
    mindspore.nn.GraphCell
    mindspore.nn.LossBase
    mindspore.nn.Optimizer

容器
-----------

.. mscnplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.CellList
    mindspore.nn.SequentialCell

封装层
-----------

.. mscnplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.DistributedGradReducer
    mindspore.nn.DynamicLossScaleUpdateCell
    mindspore.nn.FixedLossScaleUpdateCell
    mindspore.nn.ForwardValueAndGrad
    mindspore.nn.GetNextSingleOp
    mindspore.nn.MicroBatchInterleaved
    mindspore.nn.ParameterUpdate
    mindspore.nn.PipelineCell
    mindspore.nn.TimeDistributed
    mindspore.nn.TrainOneStepCell
    mindspore.nn.TrainOneStepWithLossScaleCell
    mindspore.nn.WithEvalCell
    mindspore.nn.WithGradCell
    mindspore.nn.WithLossCell

卷积神经网络层
--------------------

.. mscnplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.Conv1d
    mindspore.nn.Conv1dTranspose
    mindspore.nn.Conv2d
    mindspore.nn.Conv2dTranspose
    mindspore.nn.Conv3d
    mindspore.nn.Conv3dTranspose
    mindspore.nn.Unfold

循环神经网络层
-----------------

.. mscnplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.RNN
    mindspore.nn.RNNCell
    mindspore.nn.GRU
    mindspore.nn.GRUCell
    mindspore.nn.LSTM
    mindspore.nn.LSTMCell

嵌入层
-----------------

.. mscnplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.Embedding
    mindspore.nn.EmbeddingLookup
    mindspore.nn.MultiFieldEmbeddingLookup

非线性激活函数层
------------------

.. mscnplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.CELU
    mindspore.nn.ELU
    mindspore.nn.FastGelu
    mindspore.nn.GELU
    mindspore.nn.get_activation
    mindspore.nn.Hardtanh
    mindspore.nn.HShrink
    mindspore.nn.HSigmoid
    mindspore.nn.HSwish
    mindspore.nn.LeakyReLU
    mindspore.nn.LogSigmoid
    mindspore.nn.LogSoftmax
    mindspore.nn.LRN
    mindspore.nn.Mish
    mindspore.nn.Softsign
    mindspore.nn.PReLU
    mindspore.nn.ReLU
    mindspore.nn.ReLU6
    mindspore.nn.RReLU
    mindspore.nn.SeLU
    mindspore.nn.SiLU
    mindspore.nn.Sigmoid
    mindspore.nn.Softmin
    mindspore.nn.Softmax
    mindspore.nn.SoftShrink
    mindspore.nn.Tanh
    mindspore.nn.Tanhshrink
    mindspore.nn.Threshold

线性层
-----------------

.. mscnplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.Dense
    mindspore.nn.BiDense

Dropout层
-----------------

.. mscnplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.Dropout
    mindspore.nn.Dropout2d
    mindspore.nn.Dropout3d

归一化层
---------

.. mscnplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.BatchNorm1d
    mindspore.nn.BatchNorm2d
    mindspore.nn.BatchNorm3d
    mindspore.nn.GroupNorm
    mindspore.nn.InstanceNorm1d
    mindspore.nn.InstanceNorm2d
    mindspore.nn.InstanceNorm3d
    mindspore.nn.LayerNorm
    mindspore.nn.SyncBatchNorm

池化层
--------------

.. mscnplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.AdaptiveAvgPool1d
    mindspore.nn.AdaptiveAvgPool2d
    mindspore.nn.AdaptiveAvgPool3d
    mindspore.nn.AdaptiveMaxPool1d
    mindspore.nn.AdaptiveMaxPool2d
    mindspore.nn.AvgPool1d
    mindspore.nn.AvgPool2d
    mindspore.nn.MaxPool1d
    mindspore.nn.MaxPool2d

填充层
--------------

.. mscnplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.Pad
    mindspore.nn.ConstantPad1d
    mindspore.nn.ConstantPad2d
    mindspore.nn.ConstantPad3d
    mindspore.nn.ReflectionPad1d
    mindspore.nn.ReflectionPad2d
    mindspore.nn.ZeroPad2d

损失函数
--------

.. mscnplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.BCELoss
    mindspore.nn.BCEWithLogitsLoss
    mindspore.nn.CosineEmbeddingLoss
    mindspore.nn.CrossEntropyLoss
    mindspore.nn.DiceLoss
    mindspore.nn.FocalLoss
    mindspore.nn.HuberLoss
    mindspore.nn.L1Loss
    mindspore.nn.MSELoss
    mindspore.nn.MultiClassDiceLoss
    mindspore.nn.NLLLoss
    mindspore.nn.RMSELoss
    mindspore.nn.SampledSoftmaxLoss
    mindspore.nn.SmoothL1Loss
    mindspore.nn.SoftMarginLoss
    mindspore.nn.SoftmaxCrossEntropyWithLogits

优化器
-------

.. mscnplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.Adadelta
    mindspore.nn.Adagrad
    mindspore.nn.Adam
    mindspore.nn.AdaMax
    mindspore.nn.AdamOffload
    mindspore.nn.AdamWeightDecay
    mindspore.nn.AdaSumByDeltaWeightWrapCell
    mindspore.nn.AdaSumByGradWrapCell
    mindspore.nn.ASGD
    mindspore.nn.FTRL
    mindspore.nn.Lamb
    mindspore.nn.LARS
    mindspore.nn.LazyAdam
    mindspore.nn.Momentum
    mindspore.nn.ProximalAdagrad
    mindspore.nn.RMSProp
    mindspore.nn.Rprop
    mindspore.nn.SGD
    mindspore.nn.thor

评价指标
--------

.. mscnplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.Accuracy
    mindspore.nn.auc
    mindspore.nn.BleuScore
    mindspore.nn.ConfusionMatrix
    mindspore.nn.ConfusionMatrixMetric
    mindspore.nn.CosineSimilarity
    mindspore.nn.Dice
    mindspore.nn.F1
    mindspore.nn.Fbeta
    mindspore.nn.HausdorffDistance
    mindspore.nn.get_metric_fn
    mindspore.nn.Loss
    mindspore.nn.MAE
    mindspore.nn.MeanSurfaceDistance
    mindspore.nn.Metric
    mindspore.nn.MSE
    mindspore.nn.names
    mindspore.nn.OcclusionSensitivity
    mindspore.nn.Perplexity
    mindspore.nn.Precision
    mindspore.nn.Recall
    mindspore.nn.ROC
    mindspore.nn.RootMeanSquareDistance
    mindspore.nn.rearrange_inputs
    mindspore.nn.Top1CategoricalAccuracy
    mindspore.nn.Top5CategoricalAccuracy
    mindspore.nn.TopKCategoricalAccuracy

动态学习率
-----------

LearningRateSchedule类
^^^^^^^^^^^^^^^^^^^^^^^

本模块中的动态学习率都是LearningRateSchedule的子类，将LearningRateSchedule的实例传递给优化器。在训练过程中，优化器以当前step为输入调用该实例，得到当前的学习率。

.. code-block::

    import mindspore.nn as nn

    min_lr = 0.01
    max_lr = 0.1
    decay_steps = 4
    cosine_decay_lr = nn.CosineDecayLR(min_lr, max_lr, decay_steps)

    net = Net()
    optim = nn.Momentum(net.trainable_params(), learning_rate=cosine_decay_lr, momentum=0.9)

.. mscnplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.CosineDecayLR
    mindspore.nn.ExponentialDecayLR
    mindspore.nn.InverseDecayLR
    mindspore.nn.NaturalExpDecayLR
    mindspore.nn.PolynomialDecayLR
    mindspore.nn.WarmUpLR

Dynamic LR函数
^^^^^^^^^^^^^^

本模块中的动态学习率都是function，调用function并将结果传递给优化器。在训练过程中，优化器将result[current step]作为当前学习率。

.. code-block::

    import mindspore.nn as nn

    min_lr = 0.01
    max_lr = 0.1
    total_step = 6
    step_per_epoch = 1
    decay_epoch = 4

    lr= nn.cosine_decay_lr(min_lr, max_lr, total_step, step_per_epoch, decay_epoch)

    net = Net()
    optim = nn.Momentum(net.trainable_params(), learning_rate=lr, momentum=0.9)

.. mscnplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.cosine_decay_lr
    mindspore.nn.exponential_decay_lr
    mindspore.nn.inverse_decay_lr
    mindspore.nn.natural_exp_decay_lr
    mindspore.nn.piecewise_constant_lr
    mindspore.nn.polynomial_decay_lr
    mindspore.nn.warmup_lr

图像处理层
-----------

.. mscnplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.CentralCrop
    mindspore.nn.ImageGradients
    mindspore.nn.MSSSIM
    mindspore.nn.PSNR
    mindspore.nn.ResizeBilinear
    mindspore.nn.SSIM

矩阵处理
-----------

.. mscnplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.MatrixDiag
    mindspore.nn.MatrixDiagPart
    mindspore.nn.MatrixSetDiag

工具
-----

.. mscnplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.ClipByNorm
    mindspore.nn.Flatten
    mindspore.nn.L1Regularizer
    mindspore.nn.Norm
    mindspore.nn.OneHot
    mindspore.nn.Range
    mindspore.nn.Roll
    mindspore.nn.Tril
    mindspore.nn.Triu

数学运算
----------

.. mscnplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.Moments
