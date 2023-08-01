mindspore.nn.ConstantLR
==========================

.. py:class:: mindspore.nn.ConstantLR(optimizer, factor=1.0 / 3, total_iters=5, last_epoch=-1, verbose=False)

    将每个参数组的学习率按照衰减因子 `factor` 进行衰减，直到epoch/step数达到 `total_iters`。注意，这种衰减可能与外部对于学习率的改变同时发生。

    .. warning::
        这是一个实验性的动态学习率接口，需要和 `实验性优化器 <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.nn.html#%E5%AE%9E%E9%AA%8C%E6%80%A7%E4%BC%98%E5%8C%96%E5%99%A8>`_ 下的接口配合使用。

    参数：
        - **optimizer** (:class:`mindspore.nn.optim_ex.Optimizer`) - 优化器实例。
        - **factor** (float) - 学习率的衰减因子。 默认值：``1.0 / 3``。
        - **total_iters** (int) - 学习率进行衰减的epoch/step数，当epoch/step数达到 `total_iters`，恢复学习率。默认值：``5``.
        - **last_epoch** (int，可选) - epoch/step数。默认值：``-1``。
        - **verbose** (bool，可选) - 是否打印学习率。默认值：``False``。
