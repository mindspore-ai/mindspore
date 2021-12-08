mindspore.nn
=============

Cell
----

.. cnmsplatformautosummary::
    :toctree: nn

    mindspore.nn.Cell

Containers
-----------

.. cnmsplatformautosummary::
    :toctree: nn

    mindspore.nn.CellList
    mindspore.nn.SequentialCell

Gradient
---------

.. cnmsplatformautosummary::
    :toctree: nn

    mindspore.nn.Jvp
    mindspore.nn.Vjp

Non-linear Activations
----------------------

.. cnmsplatformautosummary::
    :toctree: nn

    mindspore.nn.FastGelu
    mindspore.nn.HSwish
    mindspore.nn.LeakyReLU
    mindspore.nn.ReLU
    mindspore.nn.ELU
    mindspore.nn.GELU
    mindspore.nn.Softmax

Utilities
---------

.. cnmsplatformautosummary::
    :toctree: nn

    mindspore.nn.Flatten
    mindspore.nn.Tril

Loss Functions
--------------

.. cnmsplatformautosummary::
    :toctree: nn

    mindspore.nn.SmoothL1Loss

Optimizer Functions
-------------------

.. cnmsplatformautosummary::
    :toctree: nn

    mindspore.nn.Adagrad
    mindspore.nn.Adam
    mindspore.nn.AdamOffload
    mindspore.nn.AdamWeightDecay
    mindspore.nn.FTRL
    mindspore.nn.LARS
    mindspore.nn.Lamb
    mindspore.nn.LazyAdam
    mindspore.nn.Momentum
    mindspore.nn.Optimizer
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
