mindspore.nn.inverse_decay_lr
=============================

.. py:class:: mindspore.nn.inverse_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch, is_stair=False)

    基于逆时衰减函数计算学习率。每个step的学习率将会被存放在一个列表中。

    对于第i步，计算decayed_learning_rate[i]的公式为：

    .. math::
        decayed\_learning\_rate[i] = learning\_rate / (1 + decay\_rate * current\_epoch / decay\_epoch)

    其中，:math:`current\_epoch=floor(\frac{i}{step\_per\_epoch})`。

    **参数：**

    - **learning_rate** (float) - 学习率的初始值。
    - **decay_rate** (float) - 衰减率。
    - **total_step** (int) - step总数。
    - **step_per_epoch** (int) - 每个epoch的step数。
    - **decay_epoch** (int) - 进行衰减的epoch数。
    - **is_stair** (bool) - 如果为True，则学习率每 `decay_epoch` 次衰减一次。默认值：False。

    **返回：**

    list[float]。列表大小为 `total_step` 。

    **异常：**

    - **TypeError:** `total_step` 或 `step_per_epoch` 或 `decay_epoch` 不是int。
    - **TypeError:** `is_stair` 不是bool。
    - **ValueError:** `learning_rate` 或 `decay_rate` 不是float。
    - **ValueError:** `learning_rate` 或 `decay_rate` 小于等于0。

    **样例：**

    >>> import mindspore.nn as nn
    >>>
    >>> learning_rate = 0.1
    >>> decay_rate = 0.5
    >>> total_step = 6
    >>> step_per_epoch = 1
    >>> decay_epoch = 1
    >>> output = nn.inverse_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch, True)
    >>> print(output)
    [0.1, 0.06666666666666667, 0.05, 0.04, 0.03333333333333333, 0.028571428571428574]
