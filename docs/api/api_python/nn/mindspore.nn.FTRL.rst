mindspore.nn.FTRL
=================

.. py:class:: mindspore.nn.FTRL(params, initial_accum=0.1, learning_rate=0.001, lr_power=-0.5, l1=0.0, l2=0.0, use_locking=False, loss_scale=1.0, weight_decay=0.0)

    FTRL算法实现。

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

    参数：
        - **params** (Union[list[Parameter], list[dict]]) - 必须是 `Parameter` 组成的列表或字典组成的列表。当列表元素是字典时，字典的键可以是"params"、"lr"、"weight_decay"、"grad_centralization"和"order_params"：

          .. include:: mindspore.nn.optim_group_param.rst

          - **lr** - 学习率当前不支持参数分组。

          .. include:: mindspore.nn.optim_group_dynamic_weight_decay.rst

          .. include:: mindspore.nn.optim_group_gc.rst

          .. include:: mindspore.nn.optim_group_order.rst

        - **initial_accum** (float) - 累加器 `m` 的初始值，必须大于等于零。默认值：0.1。
        - **learning_rate** (float) - 学习速率值必须为零或正数，当前不支持动态学习率。默认值：0.001。
        - **lr_power** (float) - 学习率的幂值，控制训练期间学习率的下降方式，必须小于或等于零。如果lr_power为零，则使用固定的学习率。默认值：-0.5。
        - **l1** (float) - l1正则化强度，必须大于等于零。默认值：0.0。
        - **l2** (float) - l2正则化强度，必须大于等于零。默认值：0.0。
        - **use_locking** (bool) - 如果为True，则更新操作使用锁保护。默认值：False。

        .. include:: mindspore.nn.optim_arg_loss_scale.rst

        - **weight_decay** (Union[float, int, Cell]) - 权重衰减（L2 penalty）。默认值：0.0。

          .. include:: mindspore.nn.optim_arg_dynamic_wd.rst

    输入：
        - **grads** (tuple[Tensor]) - 优化器中 `params` 的梯度，shape与优化器中的 `params` 相同。

    输出：
        tuple[Parameter]，更新的参数，shape与 `params` 相同。

    异常：
        - **TypeError** - `initial_accum`、`learning_rate`、`lr_power`、`l1`、`l2` 或 `loss_scale` 不是float。
        - **TypeError** - `parameters` 的元素不是Parameter或dict。
        - **TypeError** - `weight_decay` 不是float或int。
        - **TypeError** - `use_nesterov` 不是bool。
        - **ValueError** - `lr_power` 大于0。
        - **ValueError** - `loss_scale` 小于等于0。
        - **ValueError** - `initial_accum`、`l1` 或 `l2` 小于0。
