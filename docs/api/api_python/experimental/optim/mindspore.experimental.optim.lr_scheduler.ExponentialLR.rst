mindspore.experimental.optim.lr_scheduler.ExponentialLR
==========================================================

.. py:class:: mindspore.experimental.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)

    每个epoch呈指数衰减的学习率，即乘以 `gamma` 。注意，这种衰减可能与外部对于学习率的改变同时发生。

    .. warning::
        这是一个实验性的动态学习率接口，需要和 `mindspore.experimental.optim <https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.experimental.html#%E5%AE%9E%E9%AA%8C%E6%80%A7%E4%BC%98%E5%8C%96%E5%99%A8>`_ 下的接口配合使用。

    参数：
        - **optimizer** (:class:`mindspore.experimental.optim.Optimizer`) - 优化器实例。
        - **gamma** (float) -  学习率衰减的乘法因子。
        - **last_epoch** (int，可选) - 最后一个epoch的索引。默认值： ``-1``。
