mindspore.experimental.optim.lr_scheduler.ChainedScheduler
============================================================

.. py:class:: mindspore.experimental.optim.lr_scheduler.ChainedScheduler(schedulers)

    保存多个学习率调度器的链表，调用step()函数可以执行每一个学习率调度器的step()函数。

    .. warning::
        这是一个实验性的动态学习率接口，需要和 `mindspore.experimental.optim` 下的接口配合使用。

    参数：
        - **schedulers** (list[:class:`mindspore.experimental.optim.lr_scheduler.LRScheduler`]) - 学习率调度器的列表。

    .. py:method:: get_last_lr()

        返回当前使用的学习率。

    .. py:method:: step()

        顺序执行保存的学习率调度器的step()函数。
