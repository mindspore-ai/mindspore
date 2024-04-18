mindspore.experimental.optim.NAdam
===================================

.. py:class:: mindspore.experimental.optim.NAdam(params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, momentum_decay=4e-3)

    NAdam算法的实现。

    .. warning::
        这是一个实验性的优化器接口，需要和 `LRScheduler <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.experimental.html#lrscheduler%E7%B1%BB>`_ 下的动态学习率接口配合使用。

    参数：
        - **params** (Union[list(Parameter), list(dict)]) - 网络参数的列表或指定了参数组的列表。
        - **lr** (Union[int, float, Tensor], 可选) - 学习率。默认值：``2e-3``。
        - **betas** (Tuple[float, float], 可选) - 梯度及其平方的运行平均值的系数。默认值：``(0.9, 0.999)``。
        - **eps** (float, 可选) - 加在分母上的值，以确保数值稳定。必须大于0。默认值：``1e-8``。
        - **weight_decay** (float, 可选) - 权重衰减（L2 penalty）。默认值：``0.``。
        - **momentum_decay** (float, 可选) - 动量衰减系数。默认值：``4e-3``。

    输入：
        - **gradients** (tuple[Tensor]) - 网络权重的梯度。

    异常：
        - **ValueError** - 学习率不是int、float或Tensor。
        - **ValueError** - 学习率小于0。
        - **ValueError** - `eps` 小于0。
        - **ValueError** - `weight_decay` 小于0。
        - **ValueError** - `momentum_decay` 小于0。
        - **ValueError** - `betas` 内元素取值范围不在[0, 1)之间。
