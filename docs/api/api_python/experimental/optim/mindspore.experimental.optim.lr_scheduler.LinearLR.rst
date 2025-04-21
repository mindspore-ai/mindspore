mindspore.experimental.optim.lr_scheduler.LinearLR
========================================================

.. py:class:: mindspore.experimental.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0 / 3, end_factor=1.0, total_iters=5, last_epoch=-1)

    线性减小学习率乘法因子 ，并将每个参数组的学习率按照此乘法因子进行衰减，直到 `last_epoch` 数达到 `total_iters`。注意，这种衰减可能与外部对于学习率的改变同时发生。

    .. warning::
        这是一个实验性的动态学习率接口，需要和 `mindspore.experimental.optim <https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.experimental.html#%E5%AE%9E%E9%AA%8C%E6%80%A7%E4%BC%98%E5%8C%96%E5%99%A8>`_ 下的接口配合使用。

    参数：
        - **optimizer** (:class:`mindspore.experimental.optim.Optimizer`) - 优化器实例。
        - **start_factor** (float，可选) - 初始的乘法因子值，后续向 `end_factor` 进行线性变化。默认值： ``1.0 /3``。
        - **end_factor** (float，可选) - 线性变化过程结束时的乘法因子值。默认值： ``1.0``。
        - **total_iters** (int，可选) - 迭代的次数。默认值： ``5``。
        - **last_epoch** (int，可选) - 当前scheduler的 `step()` 方法的执行次数。默认值： ``-1``。

    异常：
        - **ValueError** - `start_factor` 不在(0, 1]范围内。
        - **ValueError** - `end_factor` 不在[0, 1]范围内。