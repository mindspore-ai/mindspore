mindspore.nn.SGD
================

.. py:class:: mindspore.nn.SGD(*args, **kwargs)

    随机梯度下降的实现。动量可选。

    SGD相关介绍参见 `SGD <https://en.wikipedia.org/wiki/Stochastic_gradient_dencent>`_ 。

    Nesterov动量公式参见论文 `On the importance of initialization and momentum in deep learning <http://proceedings.mlr.press/v28/sutskever13.html>`_ 。

    .. math::
            v_{t+1} = u \ast v_{t} + gradient \ast (1-dampening)

    如果nesterov为True：

    .. math::
            p_{t+1} = p_{t} - lr \ast (gradient + u \ast v_{t+1})

    如果nesterov为False：

    .. math::
            p_{t+1} = p_{t} - lr \ast v_{t+1}

    需要注意的是，对于训练的第一步 :math:`v_{t+1} = gradient`。其中，p、v和u分别表示 `parameters`、`accum` 和 `momentum`。

    .. note::

        .. include:: mindspore.nn.optim_note_weight_decay.rst

    **参数：**

    - **params** (Union[list[Parameter], list[dict]]): 当 `params` 为会更新的 `Parameter` 列表时，`params` 中的元素必须为类 `Parameter`。当 `params` 为 `dict` 列表时，"params"、"lr"、"weight_decay"、"grad_centralization"和"order_params"为可以解析的键。

      .. include:: mindspore.nn.optim_group_param.rst
      .. include:: mindspore.nn.optim_group_lr.rst
      .. include:: mindspore.nn.optim_group_weight_decay.rst
      .. include:: mindspore.nn.optim_group_gc.rst
      .. include:: mindspore.nn.optim_group_order.rst

    - **learning_rate** (Union[float, Tensor, Iterable, LearningRateSchedule]): 默认值：0.1。

      .. include:: mindspore.nn.optim_arg_dynamic_lr.rst

    - **momentum** (float): 浮点动量，必须大于等于0.0。默认值：0.0。
    - **dampening** (float): 浮点动量阻尼值，必须大于等于0.0。默认值：0.0。
    - **weight_decay** (float): 权重衰减（L2 penalty），必须大于等于0。默认值：0.0。
    - **nesterov** (bool): 启用Nesterov动量。如果使用Nesterov，动量必须为正，阻尼必须等于0.0。默认值：False。

      .. include:: mindspore.nn.optim_arg_loss_scale.rst

    **输入：**

    - **gradients** (tuple[Tensor]) - `params` 的梯度，shape与 `params` 相同。

    **输出：**

    Tensor[bool]，值为True。

    **异常：**

    **ValueError：** 动量、阻尼或重量衰减值小于0.0。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> net = Net()
    >>> # 1) 所有参数使用相同的学习率和权重衰减
    >>> optim = nn.SGD(params=net.trainable_params())
    >>>
    >>> # 2) 使用参数组并设置不同的值
    >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
    >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
    >>> group_params = [{'params': conv_params,'grad_centralization':True},
    ...                 {'params': no_conv_params, 'lr': 0.01},
    ...                 {'order_params': net.trainable_params()}]
    >>> optim = nn.SGD(group_params, learning_rate=0.1, weight_decay=0.0)
    >>> # con_params的参数将使用默认学习率0.1、默认权重衰减0.0、梯度集中度为True。
    >>> #
    >>> # no_con_params的参数将使用学习率0.01、默认权重衰减0.0、梯度集中度为False。
    >>> #
    >>> # 优化器的最终参数顺序采用'order_params'的值。
    >>>
    >>> loss = nn.SoftmaxCrossEntropyWithLogits()
    >>> model = Model(net, loss_fn=loss, optimizer=optim)
