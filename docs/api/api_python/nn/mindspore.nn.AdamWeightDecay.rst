mindspore.nn.AdamWeightDecay
===============================

.. py:class:: mindspore.nn.AdamWeightDecay(params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0)

    实现权重衰减Adam算法。

    .. math::
        \begin{array}{ll} \\
            m_{t+1} = \beta_1 * m_{t} + (1 - \beta_1) * g \\
            v_{t+1} = \beta_2 * v_{t} + (1 - \beta_2) * g * g \\
            update = \frac{m_{t+1}}{\sqrt{v_{t+1}} + eps} \\
            update =
            \begin{cases}
                update + weight\_decay * w_{t}
                    & \text{ if } weight\_decay > 0 \\
                update
                    & \text{ otherwise }
            \end{cases} \\
            w_{t+1}  = w_{t} - lr * update
        \end{array}

    :math:`m` 表示第1矩向量 `moment1` , :math:`v` 表示第2矩向量 `moment2`， :math:`g` 表示 `gradients` ，:math:`lr` 表示 `learning_rate` ，:math:`\beta_1, \beta_2` 表示 `beta1` 和 `beta2` , :math:`t` 表示当前step，:math:`w` 表示 `params`。

    .. note::
        .. include:: mindspore.nn.optim_note_loss_scale.rst
        .. include:: mindspore.nn.optim_note_weight_decay.rst

    **参数：**

    - **params** (Union[list[Parameter], list[dict]]) - 必须是 `Parameter` 组成的列表或字典组成的列表。当列表元素是字典时，字典的键可以是"params"、"lr"、"weight_decay"、和"order_params"：

      .. include:: mindspore.nn.optim_group_param.rst

      .. include:: mindspore.nn.optim_group_lr.rst

      .. include:: mindspore.nn.optim_group_weight_decay.rst

      .. include:: mindspore.nn.optim_group_order.rst


    - **learning_rate** (Union[float, Tensor, Iterable, LearningRateSchedule]): 默认值：1e-3。

      .. include:: mindspore.nn.optim_arg_dynamic_lr.rst

    - **beta1** (float)：`moment1` 的指数衰减率。参数范围（0.0,1.0）。默认值：0.9。
    - **beta2** (float)：`moment2` 的指数衰减率。参数范围（0.0,1.0）。默认值：0.999。
    - **eps** (float) - 将添加到分母中，以提高数值稳定性。必须大于0。默认值：1e-6。
    - **weight_decay** (float) - 权重衰减（L2 penalty）。必须大于等于0。默认值：0.0。

    **输入：**

    **gradients** (tuple[Tensor]) - `params` 的梯度，shape与 `params` 相同。

    **输出：**

    **tuple** [bool]，所有元素都为True。

    **异常：**

    - **TypeError** - `learning_rate` 不是int、float、Tensor、Iterable或LearningRateSchedule。
    - **TypeError** - `parameters` 的元素不是Parameter或字典。
    - **TypeError** - `beta1` 、 `beta2` 或 `eps` 不是float。
    - **TypeError** - `weight_decay` 不是float或int。
    - **ValueError** - `eps` 小于等于0。
    - **ValueError** - `beta1` 、 `beta2` 不在（0.0,1.0）范围内。
    - **ValueError** - `weight_decay` 小于0。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> net = Net()
    >>> #1) 所有参数使用相同的学习率和权重衰减
    >>> optim = nn.AdamWeightDecay(params=net.trainable_params())
    >>>
    >>> #2) 使用参数分组并设置不同的值
    >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
    >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
    >>> group_params = [{'params': conv_params, 'weight_decay': 0.01},
    ...                 {'params': no_conv_params, 'lr': 0.01},
    ...                 {'order_params': net.trainable_params()}]
    >>> optim = nn.AdamWeightDecay(group_params, learning_rate=0.1, weight_decay=0.0)
    >>> # conv_params参数组将使用优化器中的学习率0.1、该组的权重衰减0.01。
    >>> # no_conv_params参数组将使用该组的学习率0.01、优化器中的权重衰减0.0。
    >>> # 优化器按照"order_params"配置的参数顺序更新参数。
    >>>
    >>> loss = nn.SoftmaxCrossEntropyWithLogits()
    >>> model = Model(net, loss_fn=loss, optimizer=optim)
