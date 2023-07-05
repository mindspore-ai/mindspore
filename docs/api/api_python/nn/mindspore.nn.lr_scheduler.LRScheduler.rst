mindspore.nn.LRScheduler
======================================

.. py:class:: mindspore.nn.LRScheduler(optimizer, last_epoch=-1, verbose=False)

    动态学习率的基类。

    .. warning::
        这是一个实验性的动态学习率模块，需要和 `nn.optim_ex` 下的优化器配合使用。

    参数：
        - **optimizer** (:class:`mindspore.nn.optim_ex.Optimizer`) - 优化器实例。
        - **last_epoch** (int，可选) - epoch/step数。默认值： ``-1``。
        - **verbose** (bool，可选) - 是否打印学习率。默认值： ``False``。

    异常：
        - **TypeError** - `optimizer` 不是优化器。
        - **TypeError** - `last_epoch` 小于-1。
        - **ValueError** - `verbose` 不是布尔类型。

    .. py:method:: get_last_lr()

        返回当前使用的学习率。

    .. py:method:: step()

        按照定义的计算逻辑计算并修改学习率。
