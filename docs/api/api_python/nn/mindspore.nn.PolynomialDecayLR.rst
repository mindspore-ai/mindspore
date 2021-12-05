mindspore.nn.PolynomialDecayLR
====================================

.. py:class:: mindspore.nn.PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps, power, update_decay_steps=False)

    基于多项式衰减函数计算学习率。

    对于当前step，计算decayed_learning_rate[current_step]的公式为：

    .. math::
        decayed\_learning\_rate[current\_step] = (learning\_rate - end\_learning\_rate) *
        (1 - tmp\_step / tmp\_decay\_steps)^{power} + end\_learning\_rate

    其中，

    .. math::
        tmp\_step=min(current\_step, decay\_steps)

    如果 `update_decay_steps` 为true，则每 `decay_steps` 更新 `tmp_decay_step` 的值。公式为：

    .. math::
        tmp\_decay\_steps = decay\_steps * ceil(current\_step / decay\_steps)

    **参数：**

    - **learning_rate** (float) - 学习率的初始值。
    - **end_learning_rate** (float) - 学习率的最终值。
    - **decay_steps** (int) - 用于计算衰减学习率的值。
    - **power** (float) - 用于计算衰减学习率的值。该参数必须大于0。
    - **update_decay_steps** (bool) - 如果为True，则学习率每 `decay_steps` 次衰减一次。默认值：False。

    **输入：**

    - **global_step** （Tensor）：当前step数。

    **输出：**

    Tensor。当前step的学习率值, shape为 :math:`()`。

    **异常：**

    - **TypeError** - `learning_rate`, `end_learning_rate` 或 `power` 不是float。
    - **TypeError** - `decay_steps` 不是int或 `update_decay_steps` 不是bool。
    - **ValueError** - `end_learning_rate` 小于0或 `decay_steps` 小于1。
    - **ValueError** - `learning_rate` 或 `power` 小于或等于0。

    **支持平台：**

    ``Ascend`` ``GPU``

    **样例：**

    >>> learning_rate = 0.1
    >>> end_learning_rate = 0.01
    >>> decay_steps = 4
    >>> power = 0.5
    >>> global_step = Tensor(2, mstype.int32)
    >>> polynomial_decay_lr = nn.PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps, power)
    >>> result = polynomial_decay_lr(global_step)
    >>> print(result)
    0.07363961
