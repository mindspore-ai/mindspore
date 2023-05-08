mindspore.nn.warmup_lr
=======================

.. py:function:: mindspore.nn.warmup_lr(learning_rate, total_step, step_per_epoch, warmup_epoch)

    预热学习率。每个step的学习率将会被存放在一个列表中。

    对于第i步，计算warmup_learning_rate[i]的公式为：

    .. math::
        warmup\_learning\_rate[i] = learning\_rate * tmp\_epoch / warmup\_epoch

    其中 :math:`tmp\_epoch= \min(current\_epoch, warmup\_epoch),\ current\_epoch=floor(\frac{i}{step\_per\_epoch})`

    参数：
        - **learning_rate** (float) - 学习率的初始值。
        - **total_step** (int) - step总数。
        - **step_per_epoch** (int) - 每个epoch的step数。
        - **warmup_epoch** (int) - 预热学习率的epoch数。

    返回：
        list[float]。 `total_step` 表示列表的大小。

    异常：
        - **TypeError** - `learning_rate` 不是float。
        - **TypeError** - `total_step` 或 `step_per_epoch` 或 `decay_epoch` 不是int。
        - **ValueError** - `learning_rate` 小于0。
