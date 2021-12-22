mindspore.nn.natural_exp_decay_lr
=================================

.. py:class:: mindspore.nn.natural_exp_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch, is_stair=False)

    基于自然指数衰减函数计算学习率。

    对于第i步，计算decayed_learning_rate[i]的公式为：

    .. math::
        decayed\_learning\_rate[i] = learning\_rate * e^{-decay\_rate * current\_epoch}

    其中 :math:`current\_epoch=floor(\frac{i}{step\_per\_epoch})` 。

    **参数：**

    - **learning_rate** (float) - 学习率的初始值。
    - **decay_rate** (float) - 衰减率。
    - **total_step** (int) - step总数。
    - **step_per_epoch** (int) - 每个epoch的step数。
    - **decay_epoch** (int) - 用于计算衰减学习率的值。
    - **is_stair** (bool) - 如果为True，则学习率每 `decay_epoch` 次衰减一次。默认值：False。

    **返回：**

    list[float]。`total_step` 表示列表的大小。

    **样例：**

    >>> import mindspore.nn as nn
    >>>
    >>> learning_rate = 0.1
    >>> decay_rate = 0.9
    >>> total_step = 6
    >>> step_per_epoch = 2
    >>> decay_epoch = 2
    >>> output = nn.natural_exp_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch, True)
    >>> print(output)
    [0.1, 0.1, 0.1, 0.1, 0.016529888822158657, 0.016529888822158657]
