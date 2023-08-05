mindspore.experimental.optim.lr_scheduler.MultiplicativeLR
=============================================================

.. py:class:: mindspore.experimental.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda, last_epoch=-1, verbose=False)

    将每个参数组当前的学习率按照传入的 `lr_lambda` 函数乘以指定的乘法因子。当 `last_epoch = -1` 时，将学习率设置成初始学习率。

    .. warning::
        这是一个实验性的动态学习率接口，需要和 `mindspore.experimental.optim` 下的接口配合使用。

    参数：
        - **optimizer** (:class:`mindspore.experimental.optim.Optimizer`) - 优化器实例。
        - **lr_lambda** (Union(function, list)) - 一个关于epoch/step的乘法函数，或类似函数的列表，列表中每个函数对应 `optimizer.param_groups` 中的每个参数组。
        - **last_epoch** (int，可选) - epoch/step数。默认值：``-1``。
        - **verbose** (bool，可选) - 是否打印学习率。默认值： ``False``。
