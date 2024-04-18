mindspore.experimental.optim.ASGD
===================================

.. py:class:: mindspore.experimental.optim.ASGD(params, lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0.0, maximize=False)

    Averaged Stochastic Gradient Descent 算法的实现。

    .. warning::
        这是一个实验性的优化器接口，需要和 `LRScheduler <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.experimental.html#lrscheduler%E7%B1%BB>`_ 下的动态学习率接口配合使用。

    参数：
        - **params** (Union[list(Parameter), list(dict)]) - 网络参数的列表或指定了参数组的列表。
        - **lr** (Union[int, float, Tensor], 可选) - 学习率。默认值：``1e-2``。
        - **lambd** (float, 可选) - 衰减项。默认值：``1e-4``。
        - **alpha** (float, 可选) - eta更新的幂。默认值：``0.75``。
        - **t0** (float, 可选) - 开始计算平均的时刻。默认值：``1e6``。
        - **weight_decay** (float, 可选) - 权重衰减（L2 penalty）。默认值：``0.``。
        - **maximize** (bool, 可选) - 是否根据目标函数最大化网络参数。默认值：``False``。

    输入：
        - **gradients** (tuple[Tensor]) - 网络权重的梯度。

    异常：
        - **ValueError** - 学习率不是int、float或Tensor。
        - **ValueError** - 学习率小于0。
        - **ValueError** - `weight_decay` 小于0。