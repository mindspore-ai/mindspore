mindspore.nn.LazyAdam
======================

.. py:class:: mindspore.nn.LazyAdam(params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, use_locking=False, use_nesterov=False, weight_decay=0.0, loss_scale=1.0)

    Adaptive Moment Estimation (Adam)算法的实现。请参阅论文 `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_。

    当梯度稀疏时，此优化器将使用Lazy Adam算法。

    更新公式如下：

    .. math::
        \begin{array}{ll} \\
            m_{t+1} = \beta_1 * m_{t} + (1 - \beta_1) * g \\
            v_{t+1} = \beta_2 * v_{t} + (1 - \beta_2) * g * g \\
            \widehat{m_{t+1}} = \frac{m_{t+1}}{1-\beta_1^t} \\
            \widehat{v_{t+1}} = \frac{v_{t+1}}{1-\beta_2^t} \\
            w_{t+1} = w_{t} - \gamma * \frac{\widehat{m_{t+1}}}{\sqrt{\widehat{v_{t+1}}} + \epsilon}
        \end{array}

    :math:`m` 代表一阶矩变量 `moment1` ，:math:`v` 代表二阶矩变量 `moment2` ，:math:`g` 代表梯度 `gradients` ，:math:`\gamma` 代表学习率 `learning_rate` ，:math:`\beta_1, \beta_2` 代表衰减速率 `beta1` 和 `beta2` ，:math:`t` 代表当前step，:math:`beta_1^t` 和 :math:`beta_2^t` 代表 `beta1_power` 和 `beta2_power` ， :math:`w` 代表 `params` ， :math:`\epsilon` 代表 `eps` 。

    .. note::
       .. include:: mindspore.nn.optim_note_sparse.rst

       需要注意的是，梯度稀疏时该优化器只更新网络参数的当前的索引位置，稀疏行为不等同于Adam算法。如果想执行稀疏策略，那么需要把target设置为CPU。

       .. include:: mindspore.nn.optim_note_weight_decay.rst

    **参数：**

    - **params** (Union[list[Parameter], list[dict]]) - 必须是 `Parameter` 组成的列表或字典组成的列表。当列表元素是字典时，字典的键可以是"params"、"lr"、"weight_decay"、"grad_centralization"和"order_params"：

      .. include:: mindspore.nn.optim_group_param.rst
      .. include:: mindspore.nn.optim_group_lr.rst
      .. include:: mindspore.nn.optim_group_dynamic_weight_decay.rst
      .. include:: mindspore.nn.optim_group_gc.rst
      .. include:: mindspore.nn.optim_group_order.rst

    - **learning_rate** (Union[float, int, Tensor, Iterable, LearningRateSchedule]) - 默认值：1e-3。

      .. include:: mindspore.nn.optim_arg_dynamic_lr.rst

    - **beta1** (float) - `moment1` 的指数衰减率。参数范围（0.0,1.0）。默认值：0.9。
    - **beta2** (float) - `moment2` 的指数衰减率。参数范围（0.0,1.0）。默认值：0.999。
    - **eps** (float) - 将添加到分母中，以提高数值稳定性。必须大于0。默认值：1e-8。
    - **use_locking** (bool) - 是否对参数更新加锁保护。如果为True，则 `w` 、`m` 和 `v` 的Tensor更新将受到锁的保护。如果为False，则结果不可预测。默认值：False。
    - **use_nesterov** (bool) - 是否使用Nesterov Accelerated Gradient (NAG)算法更新梯度。如果为True，使用NAG更新梯度。如果为False，则在不使用NAG的情况下更新梯度。默认值：False。
    - **weight_decay** (Union[float, int, Cell]) - 权重衰减（L2 penalty）。默认值：0.0。

      .. include:: mindspore.nn.optim_arg_dynamic_wd.rst
      
    .. include:: mindspore.nn.optim_arg_loss_scale.rst

    **输入：**

    - **gradients** (tuple[Tensor]) - `params` 的梯度，shape与 `params` 相同。

    **输出：**

    Tensor[bool]，值为True。

    **异常：**

    - **TypeError** - `learning_rate` 不是int、float、Tensor、Iterable或LearningRateSchedule。
    - **TypeError** - `parameters` 的元素不是Parameter或字典。
    - **TypeError** - `beta1`、`beta2`、`eps` 或 `loss_scale` 不是float。
    - **TypeError** - `weight_decay` 不是float或int。
    - **TypeError** - `use_locking` 或 `use_nesterov` 不是bool。
    - **ValueError** - `loss_scale` 或 `eps` 小于或等于0。
    - **ValueError** - `beta1`、`beta2` 不在（0.0,1.0）范围内。
    - **ValueError** - `weight_decay` 小于0。

    .. include:: mindspore.nn.optim_target_unique_for_sparse.rst
