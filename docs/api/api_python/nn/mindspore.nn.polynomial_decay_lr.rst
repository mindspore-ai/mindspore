mindspore.nn.polynomial_decay_lr
====================================

.. py:function:: mindspore.nn.polynomial_decay_lr(learning_rate, end_learning_rate, total_step, step_per_epoch, decay_epoch, power, update_decay_epoch=False)

    基于多项式衰减函数计算学习率。每个step的学习率将会被存放在一个列表中。

    对于第i步，计算decayed_learning_rate[i]的公式为：

    .. math::
        decayed\_learning\_rate[i] = (learning\_rate - end\_learning\_rate) *
        (1 - tmp\_epoch / tmp\_decay\_epoch)^{power} + end\_learning\_rate

    其中，

    .. math::
        tmp\_epoch = \min(current\_epoch, decay\_epoch)

    .. math::
        current\_epoch=floor(\frac{i}{step\_per\_epoch})

    .. math::
        tmp\_decay\_epoch = decay\_epoch

    如果 `update_decay_epoch` 为 ``True`` ，则每个epoch更新 :math:`tmp\_decay\_epoch` 的值。公式为：

    .. math::
        tmp\_decay\_epoch = decay\_epoch * ceil(current\_epoch / decay\_epoch)

    参数：
        - **learning_rate** (float) - 学习率的初始值。
        - **end_learning_rate** (float) - 学习率的最终值。
        - **total_step** (int) - step总数。
        - **step_per_epoch** (int) - 每个epoch的step数。
        - **decay_epoch** (int) - 进行衰减的epoch数。
        - **power** (float) - 多项式的幂，必须大于0。
        - **update_decay_epoch** (bool) - 如果为 ``True`` ，则更新 `decay_epoch` 。默认值： ``False`` 。

    返回：
        list[float]。列表的大小为 `total_step`。

    异常：
        - **TypeError** - `learning_rate` 或 `end_learning_rate` 或 `power` 不是float。
        - **TypeError** - `total_step` 或 `step_per_epoch` 或 `decay_epoch` 不是int。
        - **TypeError** - `update_decay_epoch` 不是bool。
        - **ValueError** - `learning_rate` 或 `power` 小于等于0。
