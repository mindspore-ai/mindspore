mindspore.experimental.optim.lr_scheduler.LambdaLR
=====================================================

.. py:class:: mindspore.experimental.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    将每个参数组的学习率设定为初始学习率乘以指定的 `lr_lambda` 函数。当 `last_epoch = -1` 时，将学习率设置成初始学习率。

    .. warning::
        这是一个实验性的动态学习率接口，需要和 `mindspore.experimental.optim <https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.experimental.html#%E5%AE%9E%E9%AA%8C%E6%80%A7%E4%BC%98%E5%8C%96%E5%99%A8>`_ 下的接口配合使用。

    参数：
        - **optimizer** (:class:`mindspore.experimental.optim.Optimizer`) - 优化器实例。
        - **lr_lambda** (Union(function, list)) - 一个关于 `last_epoch` 的匿名函数，或类似函数的列表，列表中每个函数对应 `optimizer.param_groups` 中的每个参数组。
        - **last_epoch** (int，可选) - 当前scheduler的 `step()` 方法的执行次数。默认值：``-1``。

    异常：
        - **ValueError** - `lr_lambda` 的长度不等于参数组数目。
