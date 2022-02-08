mindspore.nn
=============

神经网络Cell。

用于构建神经网络中的预定义构建块或计算单元。

MindSpore中 `mindspore.nn` 算子与上一版本相比，新增、删除和支持平台的变化信息请参考 `API Updates <https://gitee.com/mindspore/docs/blob/master/resource/api_updates/ops_api_updates.md>`_。

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
    mindspore.nn.thor

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

本模块中的动态学习率都是LearningRateSchedule的子类，将LearningRateSchedule的实例传递给优化器。在训练过程中，优化器以当前step为输入调用该实例，得到当前的学习率。

.. code-block::

    import mindspore.nn as nn
    
    min_lr = 0.01
    max_lr = 0.1
    decay_steps = 4
    cosine_decay_lr = nn.CosineDecayLR(min_lr, max_lr, decay_steps)
    
    net = Net()
    optim = nn.Momentum(net.trainable_params(), learning_rate=cosine_decay_lr, momentum=0.9)

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
    
.. cnmsautosummary::
    :toctree: nn

    mindspore.nn.cosine_decay_lr
    mindspore.nn.exponential_decay_lr
    mindspore.nn.inverse_decay_lr
    mindspore.nn.natural_exp_decay_lr
    mindspore.nn.piecewise_constant_lr
    mindspore.nn.polynomial_decay_lr
    mindspore.nn.warmup_lr
