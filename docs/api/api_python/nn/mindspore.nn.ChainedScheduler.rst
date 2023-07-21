mindspore.nn.ChainedScheduler
=============================

.. py:class:: mindspore.nn.ChainedScheduler(schedulers)

    保存多个学习率调度器的链表，调用step()函数可以执行每一个学习率调度器的step()函数。

    .. warning::
        这是一个实验性的动态学习率接口，需要和 `实验性优化器 <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.nn.html#%E5%AE%9E%E9%AA%8C%E6%80%A7%E4%BC%98%E5%8C%96%E5%99%A8>`_ 下的接口配合使用。

    参数：
        - **schedulers** (list[:class:`mindspore.nn.LRScheduler`]) - 学习率调度器的列表。

    .. py:method:: get_last_lr()

        返回当前使用的学习率。

    .. py:method:: step()

        顺序执行保存的学习率调度器的step()函数。
