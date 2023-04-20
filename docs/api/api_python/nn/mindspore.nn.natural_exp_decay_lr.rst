mindspore.nn.natural_exp_decay_lr
=================================

.. py:function:: mindspore.nn.natural_exp_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch, is_stair=False)

    基于自然指数衰减函数计算学习率。每个step的学习率将会被存放在一个列表中。

    对于第i步，计算decayed_learning_rate[i]的公式为：

    .. math::
        decayed\_learning\_rate[i] = learning\_rate * e^{-decay\_rate * current\_epoch}

    其中 :math:`current\_epoch=floor(\frac{i}{step\_per\_epoch})` 。

    参数：
        - **learning_rate** (float) - 学习率的初始值。
        - **decay_rate** (float) - 衰减率。
        - **total_step** (int) - step总数。
        - **step_per_epoch** (int) - 每个epoch的step数。
        - **decay_epoch** (int) - 进行衰减的epoch数。
        - **is_stair** (bool) - 如果为 ``True`` ，则学习率每 `decay_epoch` 次衰减一次。默认值： ``False`` 。

    返回：
        list[float]。`total_step` 表示列表的大小。

    异常：
        - **TypeError** - `total_step` 或 `step_per_epoch` 或 `decay_epoch` 不是int。
        - **TypeError** - `is_stair` 不是bool。
        - **TypeError** - `learning_rate` 或 `decay_rate` 不是float。
        - **ValueError** - `learning_rate` 或 `decay_rate` 小于等于0。
