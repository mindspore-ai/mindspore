mindspore.nn.cosine_decay_lr
==============================

.. py:function:: mindspore.nn.cosine_decay_lr(min_lr, max_lr, total_step, step_per_epoch, decay_epoch)

    基于余弦衰减函数计算学习率。每个step的学习率将会被存放在一个列表中。

    对于第i步，计算decayed_learning_rate[i]的公式为：

    .. math::
        decayed\_learning\_rate[i] = min\_lr + 0.5 * (max\_lr - min\_lr) *
        (1 + cos(\frac{current\_epoch}{decay\_epoch}\pi))

    其中 :math:`current\_epoch=floor(\frac{i}{step\_per\_epoch})`。

    参数：
        - **min_lr** (float) - 学习率的最小值。
        - **max_lr** (float) - 学习率的最大值。
        - **total_step** (int) - step总数。
        - **step_per_epoch** (int) - 每个epoch的step数。
        - **decay_epoch** (int) - 进行衰减的epoch数。

    返回：
        list[float]。列表大小为 `total_step`。

    异常：
        - **TypeError** - `min_lr` 或 `max_lr` 不是float。
        - **TypeError** - `total_step` 或 `step_per_epoch` 或 `decay_epoch` 不是int。
        - **ValueError** - `max_lr` 不大于0或 `min_lr` 小于0。
        - **ValueError** - `total_step` 或 `step_per_epoch` 或 `decay_epoch` 小于0。
        - **ValueError** - `max_lr` 大于或等于 `min_lr`。
