mindspore.experimental.optim.Rprop
===================================

.. py:class:: mindspore.experimental.optim.Rprop(params, lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50), *, maximize=False)

    Rprop 算法的实现。

    .. warning::
        这是一个实验性的优化器接口，需要和 `LRScheduler <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.experimental.html#lrscheduler%E7%B1%BB>`_ 下的动态学习率接口配合使用。

    参数：
        - **params** (Union[list(Parameter), list(dict)]) - 网络参数的列表或指定了参数组的列表。
        - **lr** (Union[int, float, Tensor], 可选) - 学习率。默认值：``1e-2``。
        - **etas** (Tuple[float, float], 可选) - (etaminus, etaplus)，进行增大和减小的因子。默认值：``(0.5, 1.2)``。
        - **step_sizes** (Tuple[float, float], 可选) - 设定的最小步长和最大步长。默认值：``(1e-6, 50)``。

    关键字参数：
        - **maximize** (bool, 可选) - 是否根据目标函数最大化网络参数。默认值：``False``。

    输入：
        - **gradients** (tuple[Tensor]) - 网络权重的梯度。

    异常：
        - **ValueError** - 学习率不是int、float或Tensor。
        - **ValueError** - 学习率小于0。
        - **ValueError** - `etas[1]` 小于等于1.。
        - **ValueError** - `etas[0]` 不在0-1之间。
