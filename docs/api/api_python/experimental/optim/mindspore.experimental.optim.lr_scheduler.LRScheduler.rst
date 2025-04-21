mindspore.experimental.optim.lr_scheduler.LRScheduler
=======================================================

.. py:class:: mindspore.experimental.optim.lr_scheduler.LRScheduler(optimizer, last_epoch=-1)

    动态学习率的基类。

    .. warning::
        这是一个实验性的动态学习率模块，需要和 `mindspore.experimental.optim <https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.experimental.html#%E5%AE%9E%E9%AA%8C%E6%80%A7%E4%BC%98%E5%8C%96%E5%99%A8>`_ 下的接口配合使用。

    参数：
        - **optimizer** (:class:`mindspore.experimental.optim.Optimizer`) - 优化器实例。
        - **last_epoch** (int，可选) - 当前scheduler的 `step()` 方法的执行次数。默认值： ``-1``。

    异常：
        - **TypeError** - `optimizer` 不是优化器。
        - **KeyError** - `last_epoch` 不是 -1 且 ``'initial_lr'`` 不在参数组内。
        - **ValueError** - `last_epoch` 不是int类型。
        - **ValueError** - `last_epoch` 小于-1。

    .. py:method:: get_last_lr()

        返回当前使用的学习率。

    .. py:method:: step(epoch=None)

        按照定义的计算逻辑计算并修改学习率。

        参数：
            - **epoch** (int，可选) - epoch数。默认值： ``None``。


