mindspore.nn.ExponentialDecayLR
================================

.. py:class:: mindspore.nn.ExponentialDecayLR(learning_rate, decay_rate, decay_steps, is_stair=False)

    基于指数衰减函数计算学习率。

    对于当前step，计算学习率的公式为：

    .. math::
        decayed\_learning\_rate = learning\_rate * decay\_rate^{p}

    其中，

    .. math::
        p = \frac{current\_step}{decay\_steps}

    如果 `is_stair` 为True，则公式为：

    .. math::
        p = floor(\frac{current\_step}{decay\_steps})

    参数：
        - **learning_rate** (float) - 学习率的初始值。
        - **decay_rate** (float) - 衰减率。
        - **decay_steps** (int) - 进行衰减的step数。
        - **is_stair** (bool) - 如果为True，则学习率每 `decay_steps` 步衰减一次。默认值：False。

    输入：
        - **global_step** (Tensor) - 当前step数，即current_step。

    输出：
        标量Tensor。当前step的学习率值。

    异常：
        - **TypeError** - `learning_rate` 或  `decay_rate` 不是float。
        - **TypeError** - `decay_steps` 不是int或 `is_stair` 不是bool。
        - **ValueError** - `decay_steps` 小于1。
        - **ValueError** - `learning_rate` 或  `decay_rate` 小于或等于0。
