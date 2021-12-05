mindspore.nn.FTRL
=================
.. py:class:: mindspore.nn.FTRL(*args, **kwargs)

    使用ApplyFtrl算子实现FTRL算法。

    FTRL是一种在线凸优化算法，根据损失函数自适应地选择正则化函数。详见论文 `Adaptive Bound Optimization for Online Convex Optimization <https://arxiv.org/abs/1002.4908>`_。工程文档参阅 `Ad Click Prediction: a View from the Trenches <https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf>`_。


    更新公式如下：

    .. math::

        \begin{array}{ll} \\
            m_{t+1} = m_{t} + g^2 \\
            u_{t+1} = u_{t} + g  - \frac{m_{t+1}^\text{-p} - m_{t}^\text{-p}}{\alpha } * \omega_{t} \\
            \omega_{t+1} =
            \begin{cases}
                \frac{(sign(u_{t+1}) * l1 - u_{t+1})}{\frac{m_{t+1}^\text{-p}}{\alpha } + 2 * l2 }
                    & \text{ if } |u_{t+1}| > l1 \\
                0.0
                    & \text{ otherwise }
            \end{cases}\\
        \end{array}

    :math:`m` 表示累加器，:math:`g` 表示 `grads`， :math:`t` 表示当前step，:math:`u` 表示需要更新的线性系数，:math:`p` 表示 `lr_power`，:math:`\alpha` 表示 `learning_rate` ，:math:`\omega` 表示 `params` 。

    .. note::
        .. include:: mindspore.nn.optim_note_sparse.rst

        .. include:: mindspore.nn.optim_note_weight_decay.rst

   **参数：**

    - **params** (Union[list[Parameter], list[dict]]) - 必须是 `Parameter` 组成的列表或字典组成的列表。当列表元素是字典时，字典的键可以是"params"、"lr"、"weight_decay"、"grad_centralization"和"order_params"：

      .. include:: mindspore.nn.optim_group_param.rst

      - **lr** - 学习率当前不支持参数分组。

      .. include:: mindspore.nn.optim_group_weight_decay.rst

      .. include:: mindspore.nn.optim_group_gc.rst

      .. include:: mindspore.nn.optim_group_order.rst

    - **initial_accum** (float) - 累加器 `m` 的初始值，必须大于等于零。默认值：0.1。
    - **learning_rate** (float) - 学习速率值必须为零或正数，当前不支持动态学习率。默认值：0.001。
    - **lr_power** (float) - 学习率的幂值，控制训练期间学习率的下降方式，必须小于或等于零。如果lr_power为零，则使用固定的学习率。默认值：-0.5。
    - **l1** (float)：l1正则化强度，必须大于等于零。默认值：0.0。
    - **l2** (float)：l2正则化强度，必须大于等于零。默认值：0.0。
    - **use_locking** (bool) - 如果为True，则更新操作使用锁保护。默认值：False。

      .. include:: mindspore.nn.optim_arg_loss_scale.rst

    - **weight_decay** (Union[float, int]) - 要乘以权重的权重衰减值，必须为零或正值。默认值：0.0。

    **输入：**

    - **grads** (tuple[Tensor]) - 优化器中 `params` 的梯度，shape与优化器中的 `params` 相同。


    **输出：**

    tuple[Parameter]，更新的参数，shape与 `params` 相同。

    **异常：**

    - **TypeError** - `initial_accum`、`learning_rate`、`lr_power`、`l1`、`l2` 或 `loss_scale` 不是float。
    - **TypeError** - `parameters` 的元素不是Parameter或dict。
    - **TypeError** - `weight_decay` 不是float或int。
    - **TypeError** - `use_nesterov` 不是bool。
    - **ValueError** - `lr_power` 大于0。
    - **ValueError** - `loss_scale` 小于等于0。
    - **ValueError** - `initial_accum`、`l1` 或 `l2` 小于0。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> net = Net()
    >>> #1) 所有参数使用相同的学习率和权重衰减
    >>> optim = nn.FTRL(params=net.trainable_params())
    >>>
    >>> #2) 使用参数分组并设置不同的值
    >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
    >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
    >>> group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization':True},
    ...                 {'params': no_conv_params},
    ...                 {'order_params': net.trainable_params()}]
    >>> optim = nn.FTRL(group_params, learning_rate=0.1, weight_decay=0.0)
    >>> # conv_params参数组将使用优化器中的学习率0.1、该组的权重衰减0.01、该组的梯度中心化配置True。
    >>> # no_conv_params参数组使用优化器中的学习率0.1、优化器中的权重衰减0.0、梯度中心化使用默认值False。
    >>> # 优化器按照"order_params"配置的参数顺序更新参数。
    >>>
    >>>
    >>> loss = nn.SoftmaxCrossEntropyWithLogits()
    >>> model = Model(net, loss_fn=loss, optimizer=optim)


.. include:: mindspore.nn.optim_target_unique_for_sparse.rst