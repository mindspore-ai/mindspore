mindspore.experimental.optim.lr_scheduler.ChainedScheduler
============================================================

.. py:class:: mindspore.experimental.optim.lr_scheduler.ChainedScheduler(schedulers)

    保存多个学习率调度器的链表，调用 `ChainedScheduler.step()` 可以执行 `schedulers` 中每一个学习率调度器的 `step()` 函数。

    .. warning::
        这是一个实验性的动态学习率接口，需要和 `mindspore.experimental.optim <https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.experimental.html#%E5%AE%9E%E9%AA%8C%E6%80%A7%E4%BC%98%E5%8C%96%E5%99%A8>`_ 下的接口配合使用。

    参数：
        - **schedulers** (list[:class:`mindspore.experimental.optim.lr_scheduler.LRScheduler`]) - 学习率调度器的列表。

    .. py:method:: get_last_lr()

        返回当前使用的学习率。

    .. py:method:: step()

        顺序执行保存的学习率调度器的step()函数。
