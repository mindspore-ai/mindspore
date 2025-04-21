mindspore.experimental.optim.lr_scheduler.MultiStepLR
=======================================================

.. py:class:: mindspore.experimental.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)

    当epoch/step达到 `milestones` 时，将每个参数组的学习率按照乘法因子 `gamma` 进行变化。注意，这种衰减可能与外部对于学习率的改变同时发生。

    .. warning::
        这是一个实验性的动态学习率接口，需要和 `mindspore.experimental.optim <https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.experimental.html#%E5%AE%9E%E9%AA%8C%E6%80%A7%E4%BC%98%E5%8C%96%E5%99%A8>`_ 下的接口配合使用。

    参数：
        - **optimizer** (:class:`mindspore.experimental.optim.Optimizer`) - 优化器实例。
        - **milestones** (list) - 阈值列表，当 `last_epoch` 数达到阈值时将学习率乘以 `gamma`。
        - **gamma** (float，可选) - 学习率的乘法因子。默认值： ``0.1``。
        - **last_epoch** (int，可选) - 当前scheduler的 `step()` 方法的执行次数。默认值：``-1``。

    异常：
        - **TypeError** - `milestones` 不是列表。
        - **TypeError** - `milestones` 的元素不是int类型。
        - **TypeError** - `gamma` 不是float类型。