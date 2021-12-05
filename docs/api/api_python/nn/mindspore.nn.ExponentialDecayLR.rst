mindspore.nn.ExponentialDecayLR
================================

.. py:class:: mindspore.nn.ExponentialDecayLR(learning_rate, decay_rate, decay_steps, is_stair=False)

    基于指数衰减函数计算学习率。

    对于当前step，decayed_learning_rate[current_step]的计算公式为：

    .. math::
        decayed\_learning\_rate[current\_step] = learning\_rate * decay\_rate^{p}

    其中，

    .. math::
        p = \frac{current\_step}{decay\_steps}

    如果 `is_stair` 为True，则公式为：

    .. math::
        p = floor(\frac{current\_step}{decay\_steps})

    **参数：**

    - **learning_rate** (float): 学习率的初始值。
    - **decay_rate** (float): 衰减率。
    - **decay_steps** (int): 用于计算衰减学习率的值。
    - **is_stair** (bool): 如果为True，则学习率每 `decay_steps` 步衰减一次。默认值：False。

    **输入：**

    - **global_step** (Tensor) - 当前step数。

    **输出：**

    Tensor。形状为  :math:`()` 的当前step的学习率值。

    **异常：**

    - **TypeError:** `learning_rate` 或  `decay_rate` 不是float。
    - **TypeError:** `decay_steps` 不是int或 `is_stair` 不是bool。
    - **ValueError:** `decay_steps` 小于1。
    - **ValueError:** `learning_rate` 或  `decay_rate` 小于或等于0。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> learning_rate = 0.1
    >>> decay_rate = 0.9
    >>> decay_steps = 4
    >>> global_step = Tensor(2, mstype.int32)
    >>> exponential_decay_lr = nn.ExponentialDecayLR(learning_rate, decay_rate, decay_steps)
    >>> result = exponential_decay_lr(global_step)
    >>> print(result)
    0.09486833
