mindspore.experimental.optim.lr_scheduler.SequentialLR
=======================================================

.. py:class:: mindspore.experimental.optim.lr_scheduler.SequentialLR(optimizer, schedulers, milestones, last_epoch=-1)

    `SequentialLR` 接收一个将被顺序调用的学习率调度器列表 `schedulers`，及指定的间隔列表 `milestone`，`milestone` 设定了每个epoch哪个调度器被调用。

    .. warning::
        这是一个实验性的动态学习率接口，需要和 `mindspore.experimental.optim <https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.experimental.html#%E5%AE%9E%E9%AA%8C%E6%80%A7%E4%BC%98%E5%8C%96%E5%99%A8>`_ 下的接口配合使用。

    参数：
        - **optimizer** (:class:`mindspore.experimental.optim.Optimizer`) - 优化器实例。
        - **schedulers** (list[:class:`mindspore.experimental.optim.lr_scheduler.LRScheduler`]) - 被顺序执行的学习率调度器列表。
        - **milestones** (list) - 反应里程碑节点的整数列表。
        - **last_epoch** (int，可选) - 当前scheduler的 `step()` 方法的执行次数。默认值：``-1``。

    异常：
        - **ValueError** - `schedulers` 中的 `optimizer` 与传入的 `optimizer` 不同。
        - **ValueError** - `schedulers` 中的 `optimizer` 与 `schedulers[0].optimizer` 不同。
        - **ValueError** - `milestones` 的长度不等于 `schedulers` 的长度减1。

    .. py:method:: get_last_lr()

        返回当前使用的学习率。

    .. py:method:: step()

        按照定义的计算逻辑计算并修改学习率。
