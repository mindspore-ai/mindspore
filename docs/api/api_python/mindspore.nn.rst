mindspore.nn
=============

Cell
----

.. cnmsplatformautosummary::
    :toctree: nn

    mindspore.nn.Cell

容器
-----------

.. cnmsplatformautosummary::
    :toctree: nn

    mindspore.nn.CellList
    mindspore.nn.SequentialCell

卷积层
--------------------

.. cnmsplatformautosummary::
    :toctree: nn

    mindspore.nn.Conv1d
    mindspore.nn.Conv1dTranspose
    mindspore.nn.Conv2d
    mindspore.nn.Conv2dTranspose
    mindspore.nn.Conv3d
    mindspore.nn.Conv3dTranspose

Gradient
---------

.. cnmsplatformautosummary::
    :toctree: nn

    mindspore.nn.Jvp
    mindspore.nn.Vjp

非线性激活函数
----------------------

.. cnmsplatformautosummary::
    :toctree: nn

    mindspore.nn.FastGelu
    mindspore.nn.HShrink
    mindspore.nn.HSigmoid
    mindspore.nn.HSwish
    mindspore.nn.LeakyReLU
    mindspore.nn.LogSigmoid
    mindspore.nn.LogSoftmax
    mindspore.nn.ReLU
    mindspore.nn.ELU
    mindspore.nn.GELU
    mindspore.nn.Sigmoid
    mindspore.nn.Softmax
    mindspore.nn.Tanh

Utilities
---------

.. cnmsplatformautosummary::
    :toctree: nn

    mindspore.nn.Flatten
    mindspore.nn.Tril

损失函数
--------------

.. cnmsplatformautosummary::
    :toctree: nn

    mindspore.nn.L1Loss
    mindspore.nn.MSELoss
    mindspore.nn.SmoothL1Loss

Optimizer Functions
-------------------

.. cnmsplatformautosummary::
    :toctree: nn

    mindspore.nn.Optimizer
    mindspore.nn.Adagrad
    mindspore.nn.Adam
    mindspore.nn.AdamOffload
    mindspore.nn.AdamWeightDecay
    mindspore.nn.FTRL
    mindspore.nn.LARS
    mindspore.nn.Lamb
    mindspore.nn.LazyAdam
    mindspore.nn.Momentum
    mindspore.nn.ProximalAdagrad
    mindspore.nn.RMSProp
    mindspore.nn.SGD

Wrapper Functions
-----------------

.. cnmsplatformautosummary::
    :toctree: nn

    mindspore.nn.DistributedGradReducer
    mindspore.nn.DynamicLossScaleUpdateCell
    mindspore.nn.FixedLossScaleUpdateCell
    mindspore.nn.ForwardValueAndGrad
    mindspore.nn.PipelineCell
    mindspore.nn.TrainOneStepCell
    mindspore.nn.TrainOneStepWithLossScaleCell
    mindspore.nn.WithEvalCell
    mindspore.nn.WithLossCell

Math Functions
-----------------

.. cnmsplatformautosummary::
    :toctree: nn

    mindspore.nn.Moments

Metrics
--------

.. cnmsautosummary::
    :toctree: nn

    mindspore.nn.Accuracy
    mindspore.nn.F1
    mindspore.nn.Fbeta
    mindspore.nn.Loss
    mindspore.nn.MAE
    mindspore.nn.MSE
    mindspore.nn.Metric
    mindspore.nn.Precision
    mindspore.nn.Recall
    mindspore.nn.Top1CategoricalAccuracy
    mindspore.nn.Top5CategoricalAccuracy
    mindspore.nn.TopKCategoricalAccuracy
    mindspore.nn.get_metric_fn
    mindspore.nn.names
    mindspore.nn.rearrange_inputs

Dynamic Learning Rate
---------------------

LearningRateSchedule
^^^^^^^^^^^^^^^^^^^^^

.. cnmsplatformautosummary::
    :toctree: nn

    mindspore.nn.CosineDecayLR
    mindspore.nn.ExponentialDecayLR
    mindspore.nn.InverseDecayLR
    mindspore.nn.NaturalExpDecayLR
    mindspore.nn.PolynomialDecayLR
    mindspore.nn.WarmUpLR

Dynamic LR
^^^^^^^^^^

.. cnmsautosummary::
    :toctree: nn

    mindspore.nn.cosine_decay_lr
    mindspore.nn.exponential_decay_lr
    mindspore.nn.inverse_decay_lr
    mindspore.nn.natural_exp_decay_lr
    mindspore.nn.piecewise_constant_lr
    mindspore.nn.polynomial_decay_lr
    mindspore.nn.warmup_lr
