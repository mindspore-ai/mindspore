mindspore.nn.MultiStepLR
==========================

.. py:class:: mindspore.nn.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False)

    当epoch/step达到 `milestones` 时，将每个参数组的学习率按照乘法因子 `gamma` 进行变化。注意，这种衰减可能与外部对于学习率的改变同时发生。

    .. warning::
        这是一个实验性的动态学习率接口，需要和 `实验性优化器 <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.nn.html#%E5%AE%9E%E9%AA%8C%E6%80%A7%E4%BC%98%E5%8C%96%E5%99%A8>`_ 下的接口配合使用。

    参数：
        - **optimizer** (:class:`mindspore.nn.optim_ex.Optimizer`) - 优化器实例。
        - **milestones** (list) - 表示epoch/step阈值的列表，为递增序列，当epoch/step数达到阈值时将学习率乘以 `gamma`。
        - **gamma** (float，可选) - 学习率的乘法因子。默认值： ``0.1``。
        - **last_epoch** (int，可选) - epoch/step数。默认值：``-1``。
        - **verbose** (bool，可选) - 是否打印学习率。默认值： ``False``。
