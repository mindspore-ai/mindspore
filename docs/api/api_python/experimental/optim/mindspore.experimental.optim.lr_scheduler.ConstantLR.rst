mindspore.experimental.optim.lr_scheduler.ConstantLR
=======================================================

.. py:class:: mindspore.experimental.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0 / 3, total_iters=5, last_epoch=-1)

    将每个参数组的学习率按照衰减因子 `factor` 进行衰减，直到 `last_epoch` 达到 `total_iters`。注意，这种衰减可能与外部对于学习率的改变同时发生。

    .. warning::
        这是一个实验性的动态学习率接口，需要和 `mindspore.experimental.optim <https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.experimental.html#%E5%AE%9E%E9%AA%8C%E6%80%A7%E4%BC%98%E5%8C%96%E5%99%A8>`_ 下的接口配合使用。

    参数：
        - **optimizer** (:class:`mindspore.experimental.optim.Optimizer`) - 优化器实例。
        - **factor** (float，可选) - 学习率的衰减因子。 默认值：``1.0 / 3``。
        - **total_iters** (int，可选) - 学习率进行衰减的执行次数，当 `last_epoch` 数达到 `total_iters`，恢复学习率。默认值：``5``.
        - **last_epoch** (int，可选) - 当前scheduler的 `step()` 方法的执行次数。默认值：``-1``。
