mindspore.nn.WarmUpLR
======================

.. py:class:: mindspore.nn.WarmUpLR(learning_rate, warmup_steps)

    预热学习率。

    对于当前step，计算学习率的公式为：

    .. math::
        warmup\_learning\_rate = learning\_rate * tmp\_step / warmup\_steps

    其中，

    .. math::
        tmp\_step= \min(current\_step, warmup\_steps)

    参数：
        - **learning_rate** (float) - 学习率的初始值。
        - **warmup_steps** (int) - 学习率warmup的step数。

    输入：
        - **global_step** (Tensor) - 当前step数，即current_step，shape为 :math:`()`。

    输出：
        标量Tensor。当前step的学习率值。

    异常：
        - **TypeError** - `learning_rate` 不是float。
        - **TypeError** - `warmup_steps` 不是int。
        - **ValueError** - `warmup_steps` 小于1。
        - **ValueError** - `learning_rate` 小于或等于0。
