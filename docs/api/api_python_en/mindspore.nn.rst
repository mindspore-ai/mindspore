mindspore.nn
=============

Neural Network Cell

For building predefined building blocks or computational units in neural networks.

Compared with the previous version, the added, deleted and supported platforms change information of `mindspore.nn` operators in MindSpore, please refer to the link `<https://gitee.com/mindspore/docs/blob/master/resource/api_updates/nn_api_updates.md>`_.

Basic Building Block
--------------------

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.Cell
    mindspore.nn.GraphCell
    mindspore.nn.LossBase
    mindspore.nn.Optimizer

Container
---------

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.CellList
    mindspore.nn.SequentialCell

Encapsulation Layer
-------------------

.. msplatformautosummary::
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

Convolutional Neural Network Layer
----------------------------------

.. msplatformautosummary::
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

Recurrent Neural Network Layer
------------------------------

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.RNN
    mindspore.nn.RNNCell
    mindspore.nn.GRU
    mindspore.nn.GRUCell
    mindspore.nn.LSTM
    mindspore.nn.LSTMCell
    
Embedding Layer
---------------

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.Embedding
    mindspore.nn.EmbeddingLookup
    mindspore.nn.MultiFieldEmbeddingLookup

Nonlinear Activation Function Layer
-----------------------------------

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.CELU
    mindspore.nn.ELU
    mindspore.nn.FastGelu
    mindspore.nn.GELU
    mindspore.nn.HShrink
    mindspore.nn.HSigmoid
    mindspore.nn.HSwish
    mindspore.nn.LeakyReLU
    mindspore.nn.LogSigmoid
    mindspore.nn.LogSoftmax
    mindspore.nn.PReLU
    mindspore.nn.ReLU
    mindspore.nn.ReLU6
    mindspore.nn.Sigmoid
    mindspore.nn.Softmax
    mindspore.nn.SoftShrink
    mindspore.nn.Tanh
    
Linear Layer
------------

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.Dense

Dropout Layer
-------------

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.Dropout

Normalization Layer
-------------------

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.BatchNorm1d
    mindspore.nn.BatchNorm2d
    mindspore.nn.BatchNorm3d
    mindspore.nn.GlobalBatchNorm
    mindspore.nn.GroupNorm
    mindspore.nn.InstanceNorm2d
    mindspore.nn.LayerNorm
    mindspore.nn.SyncBatchNorm

Pooling Layer
-------------

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.AvgPool1d
    mindspore.nn.AvgPool2d
    mindspore.nn.MaxPool1d
    mindspore.nn.MaxPool2d

Filling Layer
-------------

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.Pad

Loss Function
-------------

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.BCELoss
    mindspore.nn.BCEWithLogitsLoss
    mindspore.nn.CosineEmbeddingLoss
    mindspore.nn.DiceLoss
    mindspore.nn.FocalLoss
    mindspore.nn.L1Loss
    mindspore.nn.MSELoss
    mindspore.nn.MultiClassDiceLoss
    mindspore.nn.RMSELoss
    mindspore.nn.SampledSoftmaxLoss
    mindspore.nn.SmoothL1Loss
    mindspore.nn.SoftMarginLoss
    mindspore.nn.SoftmaxCrossEntropyWithLogits

Optimizer
---------

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.Adagrad
    mindspore.nn.Adam
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

Evaluation Metrics
------------------

.. msplatformautosummary::
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

Dynamic Learning Rate
---------------------

LearningRateSchedule Class
^^^^^^^^^^^^^^^^^^^^^^^^^^

The dynamic learning rates in this module are all subclasses of LearningRateSchedule. Pass the instance of
LearningRateSchedule to an optimizer. During the training process, the optimizer calls the instance taking current step
as input to get the current learning rate.

.. code-block::

    import mindspore.nn as nn
    
    min_lr = 0.01
    max_lr = 0.1
    decay_steps = 4
    cosine_decay_lr = nn.CosineDecayLR(min_lr, max_lr, decay_steps)
    
    net = Net()
    optim = nn.Momentum(net.trainable_params(), learning_rate=cosine_decay_lr, momentum=0.9)

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.CosineDecayLR
    mindspore.nn.ExponentialDecayLR
    mindspore.nn.InverseDecayLR
    mindspore.nn.NaturalExpDecayLR
    mindspore.nn.PolynomialDecayLR
    mindspore.nn.WarmUpLR

Dynamic LR Function
^^^^^^^^^^^^^^^^^^^

The dynamic learning rates in this module are all functions. Call the function and pass the result to an optimizer.
During the training process, the optimizer takes result[current step] as current learning rate.

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
    
.. msplatformautosummary::
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

Sparse Layer
------------

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.SparseTensorDenseMatmul
    mindspore.nn.SparseToDense

Image Processing Layer
----------------------

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.CentralCrop
    mindspore.nn.ImageGradients
    mindspore.nn.MSSSIM
    mindspore.nn.PSNR
    mindspore.nn.ResizeBilinear
    mindspore.nn.SSIM

Matrix Processing
-----------------

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.MatrixDiag
    mindspore.nn.MatrixDiagPart
    mindspore.nn.MatrixSetDiag

Tools
-----

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.ClipByNorm
    mindspore.nn.Flatten
    mindspore.nn.get_activation
    mindspore.nn.L1Regularizer
    mindspore.nn.Norm
    mindspore.nn.OneHot
    mindspore.nn.Range
    mindspore.nn.Roll
    mindspore.nn.Tril
    mindspore.nn.Triu

Mathematical Operations
-----------------------

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.MatMul
    mindspore.nn.Moments
    mindspore.nn.ReduceLogSumExp

Gradient
--------

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindspore.nn.Jvp
    mindspore.nn.Vjp
