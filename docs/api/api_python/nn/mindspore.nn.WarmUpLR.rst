mindspore.nn.WarmUpLR
======================

.. py:class:: mindspore.nn.WarmUpLR(learning_rate, warmup_steps)

    学习率热身。

    对于当前step，计算warmup_learning_rate[current_step]的公式为：

    .. math::
        warmup\_learning\_rate[current\_step] = learning\_rate * tmp\_step / warmup\_steps

    其中，

    .. math:
        tmp\_step=min(current\_step, warmup\_steps)

    **参数：**

    - **learning_rate** (float): 学习率的初始值。
    - **warmup_steps** (int): 学习率warmup的step数。

    **输入：**

    - **global_step** (Tensor) - 当前step数。

    **输出：**

    Tensor。形状为  :math:`()` 的当前step的学习率值。

    **异常：**

    - **TypeError** - `learning_rate` 不是float。
    - **TypeError** - `warmup_steps` 不是int。
    - **ValueError** - `warmup_steps` 小于1。
    - **ValueError** - `learning_rate` 小于或等于0。

    **支持平台：**

    ``Ascend`` ``GPU``

    **样例：**

    >>> learning_rate = 0.1
    >>> warmup_steps = 2
    >>> global_step = Tensor(2, mstype.int32)
    >>> warmup_lr = nn.WarmUpLR(learning_rate, warmup_steps)
    >>> result = warmup_lr(global_step)
    >>> print(result)
    0.1
