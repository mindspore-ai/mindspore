mindspore.nn.optim_ex.SGD
==========================

.. py:class:: mindspore.nn.optim_ex.SGD(params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False, *, maximize=False)

    随机梯度下降算法。

    .. math::
            v_{t+1} = u \ast v_{t} + gradient \ast (1-dampening)

    如果nesterov为True：

    .. math::
            p_{t+1} = p_{t} - lr \ast (gradient + u \ast v_{t+1})

    如果nesterov为False：

    .. math::
            p_{t+1} = p_{t} - lr \ast v_{t+1}

    需要注意的是，对于训练的第一步 :math:`v_{t+1} = gradient`。其中，p、v和u分别表示 `parameters`、`accum` 和 `momentum`。

    .. warning::
        这是一个实验性的优化器接口，需要和 `nn.lr_scheduler` 下的动态学习率接口配合使用。

    参数：
        - **params** (Union[list(Parameter), list(dict)]) - 网络参数的列表或指定了参数组的列表。
        - **lr** (Union[int, float, Tensor]) - 学习率。
        - **momentum** (Union[int, float], 可选) - 动量值。默认值：``0``。
        - **weight_decay** (float, 可选) - 权重衰减（L2 penalty），必须大于等于0。默认值：``0.0``。
        - **dampening** (Union[int, float], 可选) - 动量的阻尼值。默认值：``0``。
        - **nesterov** (bool, 可选) - 启用Nesterov动量。如果使用Nesterov，动量必须为正，阻尼必须等于0.0。默认值：``False``。

    关键字参数：
        - **maximize** (bool, 可选) - 是否根据目标函数最大化网络参数。默认值：``False``。

    输入：
        - **gradients** (tuple[Tensor]) - 网络权重的梯度。

    异常：
        - **ValueError** - 学习率不是int、float或Tensor。
        - **ValueError** - 学习率小于0。
        - **ValueError** - ``momentum`` 和 ``weight_decay`` 值小于0.0。
        - **ValueError** - ``momentum``, ``dampening`` 和 ``weight_decay`` 不是int或float。
        - **ValueError** - ``nesterov`` 和 ``maximize`` 不是布尔类型。
        - **ValueError** - ``nesterov`` 为True时, ``momentum`` 不为正或 ``dampening`` 不为0。