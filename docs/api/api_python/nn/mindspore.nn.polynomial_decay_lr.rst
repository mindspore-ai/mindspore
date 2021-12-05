mindspore.nn.polynomial_decay_lr
====================================

.. py:class:: mindspore.nn.polynomial_decay_lr(learning_rate, end_learning_rate, total_step, step_per_epoch, decay_epoch, power, update_decay_epoch=False)

    基于多项式衰减函数计算学习率。

    对于第i步，计算decayed_learning_rate[i]的公式为：

    .. math::
        decayed\_learning\_rate[i] = (learning\_rate - end\_learning\_rate) *
        (1 - tmp\_epoch / tmp\_decay\_epoch)^{power} + end\_learning\_rate

    其中，

    .. math::
        tmp\_epoch = min(current\_epoch, decay\_epoch)

    .. math::
        current\_epoch=floor(\frac{i}{step\_per\_epoch})

    .. math::
        tmp\_decay\_epoch = decay\_epoch

    如果 `update_decay_epoch` 为True，则每个epoch更新 `tmp_decay_epoch` 的值。公式为：

    .. math::
        tmp\_decay\_epoch = decay\_epoch * ceil(current\_epoch / decay\_epoch)

    **参数：**

    - **learning_rate** (float) - 学习率的初始值。
    - **end_learning_rate** (float) - 学习率的最终值。
    - **total_step** (int) - step总数。
    - **step_per_epoch** (int) - 每个epoch的step数。
    - **decay_epoch** (int) - 用于计算衰减学习率的值。
    - **power** (float) - 用于计算衰减学习率的值。该参数必须大于0。
    - **update_decay_epoch** (bool) - 如果为True，则更新 `decay_epoch` 。默认值：False。

    **返回：**

    list[float]。列表的大小为 `total_step`。

    **样例：**

    >>> learning_rate = 0.1
    >>> end_learning_rate = 0.01
    >>> total_step = 6
    >>> step_per_epoch = 2
    >>> decay_epoch = 2
    >>> power = 0.5
    >>> r = polynomial_decay_lr(learning_rate, end_learning_rate, total_step, step_per_epoch, decay_epoch, power)
    >>> print(r)
    [0.1, 0.1, 0.07363961030678928, 0.07363961030678928, 0.01, 0.01]
