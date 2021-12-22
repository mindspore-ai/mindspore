mindspore.nn.warmup_lr
=======================

.. py:function:: mindspore.nn.warmup_lr(learning_rate, total_step, step_per_epoch, warmup_epoch)

    预热学习率。

    对于第i步，计算warmup_learning_rate[i]的公式为：

    .. math::
        warmup\_learning\_rate[i] = learning\_rate * tmp\_epoch / warmup\_epoch

    其中 :math:`tmp\_epoch=min(current\_epoch, warmup\_epoch),\ current\_epoch=floor(\frac{i}{step\_per\_epoch})`

    **参数：**

    - **learning_rate** (float) - 学习率的初始值。
    - **total_step** (int) - step总数。
    - **step_per_epoch** (int) - 每个epoch的step数。
    - **warmup_epoch** (int) - 预热学习率的epoch数。

    **返回：**

    list[float]。 `total_step` 表示列表的大小。

    **样例：**

    >>> import mindspore.nn as nn
    >>>
    >>> learning_rate = 0.1
    >>> total_step = 6
    >>> step_per_epoch = 2
    >>> warmup_epoch = 2
    >>> output = nn.warmup_lr(learning_rate, total_step, step_per_epoch, warmup_epoch)
    >>> print(output)
    [0.0, 0.0, 0.05, 0.05, 0.1, 0.1]
