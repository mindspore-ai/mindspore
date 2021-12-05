mindspore.nn.inverse_decay_lr
=============================

.. py:class:: mindspore.nn.inverse_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch, is_stair=False)

    基于逆时间衰减函数计算学习率。

    对于第i步，计算decayed_learning_rate[i]的公式为：

    .. math::
        decayed\_learning\_rate[i] = learning\_rate / (1 + decay\_rate * current\_epoch / decay\_epoch)

    其中，:math:`current\_epoch=floor(\frac{i}{step\_per\_epoch})`。

    **参数：**

    - **learning_rate** (float) - 学习率的初始值。
    - **decay_rate** (float) - 衰减率。
    - **total_step** (int) - step总数。
    - **step_per_epoch** (int) - 每个epoch的step数。
    - **decay_epoch** (int) - 用于计算衰减学习率的值。
    - **is_stair** (bool) - 如果为True，则学习率每 `decay_epoch` 次衰减一次。默认值：False。

    **返回：**

    list[float]。列表大小为 `total_step` 。

    **样例：**

    >>> learning_rate = 0.1
    >>> decay_rate = 0.5
    >>> total_step = 6
    >>> step_per_epoch = 1
    >>> decay_epoch = 1
    >>> output = inverse_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch, True)
    >>> print(output)
    [0.1, 0.06666666666666667, 0.05, 0.04, 0.03333333333333333, 0.028571428571428574]
