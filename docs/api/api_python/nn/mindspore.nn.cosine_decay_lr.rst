mindspore.nn.cosine_decay_lr
==============================

.. py:class:: mindspore.nn.cosine_decay_lr(min_lr, max_lr, total_step, step_per_epoch, decay_epoch)

    基于余弦衰减函数计算学习率。

    对于第i步，计算decayed_learning_rate[i]的公式为：

    .. math::
        decayed\_learning\_rate[i] = min\_lr + 0.5 * (max\_lr - min\_lr) *
        (1 + cos(\frac{current\_epoch}{decay\_epoch}\pi))

    其中 :math:`current\_epoch=floor(\frac{i}{step\_per\_epoch})`。

   **参数：**

    - **min_lr** (float) - 学习率的最小值。
    - **max_lr** (float) - 学习率的最大值。
    - **total_step** (int) - step总数。
    - **step_per_epoch** (int) - 每个epoch的step数。
    - **decay_epoch** (int) - 用于计算衰减学习率的值。

   **返回：**

    list[float]。列表大小为 `total_step`。

   **样例：**

    >>> min_lr = 0.01
    >>> max_lr = 0.1
    >>> total_step = 6
    >>> step_per_epoch = 2
    >>> decay_epoch = 2
    >>> output = cosine_decay_lr(min_lr, max_lr, total_step, step_per_epoch, decay_epoch)
    >>> print(output)
    [0.1, 0.1, 0.05500000000000001, 0.05500000000000001, 0.01, 0.01]
