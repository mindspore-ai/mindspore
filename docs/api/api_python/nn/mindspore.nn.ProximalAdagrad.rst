mindspore.nn.ProximalAdagrad
==============================

.. py:class:: mindspore.nn.ProximalAdagrad(*args, **kwargs)

    使用ApplyProximalAdagrad算子实现ProximalAdagrad算法。

    ProximalAdagrad用于在线学习和随机优化。
    请参阅论文 `Efficient Learning using Forward-Backward Splitting <http://papers.nips.cc//paper/3793-efficient-learning-using-forward-backward-splitting.pdf>`_。

    .. math::
        accum_{t+1} = accum_{t} + grad * grad

    .. math::
        \text{prox_v} = var_{t} - lr * grad * \frac{1}{\sqrt{accum_{t+1}}}

    .. math::
        var_{t+1} = \frac{sign(\text{prox_v})}{1 + lr * l2} * \max(\left| \text{prox_v} \right| - lr * l1, 0)

    其中，grad、lr、var、accum和t分别表示 `grads`, `learning_rate`, `params` 、累加器和当前step。

    .. note::
        .. include:: mindspore.nn.optim_note_sparse.rst

        .. include:: mindspore.nn.optim_note_weight_decay.rst

    **参数：**

    - **param** (Union[list[Parameter], list[dict]]) - 必须是 `Parameter` 组成的列表或字典组成的列表。当列表元素是字典时，字典的键可以是"params"、"lr"、"weight_decay"、"grad_centralization"和"order_params"：

      .. include:: mindspore.nn.optim_group_param.rst
      .. include:: mindspore.nn.optim_group_lr.rst
      .. include:: mindspore.nn.optim_group_weight_decay.rst
      .. include:: mindspore.nn.optim_group_gc.rst
      .. include:: mindspore.nn.optim_group_order.rst

    - **accum** (float) - 累加器 `accum` 的初始值，起始值必须为零或正值。默认值：0.1。

    - **learning_rate** (Union[float, Tensor, Iterable, LearningRateSchedule]): 默认值：1e-3。

      .. include:: mindspore.nn.optim_arg_dynamic_lr.rst

    - **l1** (float):l1正则化强度，必须大于或等于零。默认值：0.0。
    - **l2** (float):l2正则化强度，必须大于或等于零。默认值：0.0。
    - **use_locking** (bool) - 如果为True，则更新操作使用锁保护。默认值：False。

      .. include:: mindspore.nn.optim_arg_loss_scale.rst

    - **weight_decay** (Union[float, int]) - 要乘以权重的权重衰减值，必须为零或正值。默认值：0.0。

    **输入：**

    - **grads** (tuple[Tensor]) - 优化器中 `params` 的梯度，shape与优化器中的 `params` 相同。

    **输出：**

    Tensor[bool]，值为True。

    **异常：**

    - **TypeError** - `learning_rate` 不是int、float、Tensor、Iterable或LearningRateSchedule。
    - **TypeError** - `parameters` 的元素不是Parameter或字典。
    - **TypeError** - `accum`、`l1`、`l2` 或 `loss_scale` 不是float。
    - **TypeError** - `weight_decay` 不是float或int。
    - **ValueError** - `loss_scale` 小于或等于0。
    - **ValueError** - `accum`、`l1`、`l2` 或 `weight_decay` 小于0。

    **支持平台：**

    ``Ascend``

    **样例：**

    >>> net = Net()
    >>> #1) 所有参数使用相同的学习率和权重衰减
    >>> optim = nn.ProximalAdagrad(params=net.trainable_params())
    >>>
    >>> #2) 使用参数组并设置不同的值
    >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
    >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
    >>> group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization':True},
    ...                 {'params': no_conv_params, 'lr': 0.01},
    ...                 {'order_params': net.trainable_params()}]
    >>> optim = nn.ProximalAdagrad(group_params, learning_rate=0.1, weight_decay=0.0)
    >>> # conv_params参数组将使用优化器中的学习率0.1、该组的权重衰减0.01、该组的梯度中心化配置True。
    >>> # no_conv_params参数组将使用该组的学习率0.01、优化器中的权重衰减0.0、梯度中心化使用默认值False。
    >>> # 优化器按照"order_params"配置的参数顺序更新参数。
    >>>
    >>> loss = nn.SoftmaxCrossEntropyWithLogits()
    >>> model = Model(net, loss_fn=loss, optimizer=optim)


.. include:: mindspore.nn.optim_target_unique_for_sparse.rst
