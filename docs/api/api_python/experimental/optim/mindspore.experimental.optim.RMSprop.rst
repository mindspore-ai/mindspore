mindspore.experimental.optim.RMSprop
======================================

.. py:class:: mindspore.experimental.optim.RMSprop(params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0, centered=False, maximize=False)

    RMSprop 算法的实现。

    .. warning::
        这是一个实验性的优化器接口，需要和 `LRScheduler <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.experimental.html#lrscheduler%E7%B1%BB>`_ 下的动态学习率接口配合使用。

    参数：
        - **params** (Union[list(Parameter), list(dict)]) - 网络参数的列表或指定了参数组的列表。
        - **lr** (Union[int, float, Tensor], 可选) - 学习率。默认值：``1e-2``。
        - **alpha** (float, 可选) - 平滑常数。默认值：``0.99``。
        - **eps** (float, 可选) - 加在分母上的值，以确保数值稳定。必须大于0。默认值：``1e-8``。
        - **weight_decay** (float, 可选) - 权重衰减（L2 penalty）。默认值：``0.``。
        - **momentum** (float, 可选) - 动量系数。默认值：``0.``。
        - **centered** (bool, 可选) - 如果为 ``True``，则计算centered RMSProp，梯度通过其方差进行归一化。默认值：``False``。
        - **maximize** (bool, 可选) - 是否根据目标函数最大化网络参数。默认值：``False``。

    输入：
        - **gradients** (tuple[Tensor]) - 网络权重的梯度。

    异常：
        - **ValueError** - 学习率不是int、float或Tensor。
        - **ValueError** - 学习率小于0。
        - **ValueError** - `momentum` 小于0。
        - **ValueError** - `alpha` 小于0。
        - **ValueError** - `eps` 小于0。
        - **ValueError** - `weight_decay` 小于0。
