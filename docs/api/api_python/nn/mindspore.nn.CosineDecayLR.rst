mindspore.nn.CosineDecayLR
===========================

.. py:class:: mindspore.nn.CosineDecayLR(min_lr, max_lr, decay_steps)

    基于余弦衰减函数计算学习率。

    对于当前step，计算学习率的公式为：

    .. math::
        decayed\_learning\_rate = &min\_lr + 0.5 * (max\_lr - min\_lr) *\\
        &(1 + cos(\frac{current\_step}{decay\_steps}\pi))


    **参数：**

    - **min_lr** (float): 学习率的最小值。
    - **max_lr** (float): 学习率的最大值。
    - **decay_steps** (int): 进行衰减的step数。

    **输入：**

    - **global_step** (Tensor) - 当前step数。

    **输出：**

    标量Tensor。当前step的学习率值。

    **异常：**

    - **TypeError:** `min_lr` 或  `max_lr` 不是float。
    - **TypeError:** `decay_steps` 不是整数。
    - **ValueError:** `min_lr` 小于0或 `decay_steps` 小于1。
    - **ValueError:** `max_lr` 小于或等于0。

    **支持平台：**

    ``Ascend`` ``GPU``

    **样例：**

    >>> import mindspore
    >>> from mindspore import Tensor, nn
    >>>
    >>> min_lr = 0.01
    >>> max_lr = 0.1
    >>> decay_steps = 4
    >>> global_steps = Tensor(2, mindspore.int32)
    >>> cosine_decay_lr = nn.CosineDecayLR(min_lr, max_lr, decay_steps)
    >>> result = cosine_decay_lr(global_steps)
    >>> print(result)
    0.055
