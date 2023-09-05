mindspore.nn.AdaMax
===================

.. py:class:: mindspore.nn.AdaMax(params, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-08, weight_decay=0.0, loss_scale=1.0)

    AdaMax算法是基于无穷范数的Adam的一种变体。

    AdaMax算法详情请参阅论文 `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_。

    公式如下：
    
    .. math::
        \begin{array}{ll} \\
            m_{t+1} = \beta_1 * m_{t} + (1 - \beta_1) * g \\
            v_{t+1} = \max(\beta_2 * v_{t}, \left| g \right|) \\
            w = w - \frac{l}{1 - \beta_1^{t+1}} * \frac{m_{t+1}}{v_{t+1} + \epsilon}
        \end{array}

    :math:`m` 代表第一个动量矩阵，:math:`v` 代表第二个动量矩阵，:math:`g` 代表梯度 `gradients` ，:math:`\beta_1, \beta_2` 代表衰减速率 `beta1` 和 `beta2` ，:math:`t` 代表当前step，:math:`beta_1^t` 代表 `beta1` 的t次方 ， :math:`l` 代表学习率 `learning_rate` ，:math:`w` 代表 `params` ， :math:`\epsilon` 代表 `eps` 。

    .. note::

        .. include:: mindspore.nn.optim_note_weight_decay.rst

    参数：
        - **params** (Union[list[Parameter], list[dict]]) - 必须是 `Parameter` 组成的列表或字典组成的列表。当列表元素是字典时，字典的键可以是"params"、"lr"、"weight_decay"、"grad_centralization"和"order_params"：

          .. include:: mindspore.nn.optim_group_param.rst
          .. include:: mindspore.nn.optim_group_lr.rst
          .. include:: mindspore.nn.optim_group_dynamic_weight_decay.rst
          .. include:: mindspore.nn.optim_group_gc.rst
          .. include:: mindspore.nn.optim_group_order.rst

        - **learning_rate** (Union[float, int, Tensor, Iterable, LearningRateSchedule]) - 默认值： ``0.001`` 。

          .. include:: mindspore.nn.optim_arg_dynamic_lr.rst

        - **beta1** (float) - 第一个动量矩阵的指数衰减率。参数范围（0.0,1.0）。默认值： ``0.9`` 。
        - **beta2** (float) - 第二个动量矩阵的指数衰减率。参数范围（0.0,1.0）。默认值： ``0.999``。
        - **eps** (float) - 加在分母上的值，以确保数值稳定。必须大于0。默认值： ``1e-08`` 。
        - **weight_decay** (Union[float, int, Cell]) - 权重衰减（L2 penalty）。默认值： ``0.0`` 。

          .. include:: mindspore.nn.optim_arg_dynamic_wd.rst

        .. include:: mindspore.nn.optim_arg_loss_scale.rst

    输入：
        - **gradients** (tuple[Tensor]) - `params` 的梯度，形状（shape）与 `params` 相同。

    输出：
        Tensor[bool]，值为True。

    异常：
        - **TypeError** - `learning_rate` 不是int、float、Tensor、iterable或LearningRateSchedule。
        - **TypeError** - `parameters` 的元素不是Parameter或字典。
        - **TypeError** - `beta1` 、`beta2` 、 `eps` 或 `loss_scale` 不是float。
        - **TypeError** - `weight_decay` 不是float或int。
        - **ValueError** - `loss_scale` 或 `eps` 小于或等于0。
        - **ValueError** - `beta1` 、`beta2` 不在（0.0,1.0）范围内。
        - **ValueError** - `weight_decay` 小于0。
