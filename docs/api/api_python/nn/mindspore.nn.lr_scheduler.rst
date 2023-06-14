mindspore.nn.LRScheduler
==========================

.. py:class:: mindspore.nn.LRScheduler(optimizer, last_epoch=-1, verbose=False)

    动态学习率的基类。

    对于当前step，计算学习率的公式为：

    .. math::
        decayed\_learning\_rate = &min\_lr + 0.5 * (max\_lr - min\_lr) *\\
        &(1 + cos(\frac{current\_step}{decay\_steps}\pi))

    参数：
        - **optimizer** (Optimizer) - 优化器实例。
        - **last_epoch** (int) - epoch/step数。默认值：-1。
        - **verbose** (bool) - 是否打印学习率. 默认值：False。

    异常：
        - **TypeError** - `optimizer` 不是优化器。
        - **TypeError** - `last_epoch` 小于-1。
        - **ValueError** - `verbose` 不是布尔类型。

    .. py:method:: step()

        按照定义的计算逻辑计算并修改学习率。


    .. py:method:: get_last_lr()

        返回当前使用的学习率。
