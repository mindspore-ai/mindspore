mindspore.nn.Momentum
======================

.. py:class:: mindspore.nn.Momentum(params, learning_rate, momentum, weight_decay=0.0, loss_scale=1.0, use_nesterov=False)

    Momentum算法的实现。

    有关更多详细信息，请参阅论文 `On the importance of initialization and momentum in deep learning <https://dl.acm.org/doi/10.5555/3042817.3043064>`_。

    .. math::
        v_{t+1} = v_{t} \ast u + grad

    如果 `use_nesterov` 为True：

    .. math::
        p_{t+1} =  p_{t} - (grad \ast lr + v_{t+1} \ast u \ast lr)

    如果 `use_nesterov` 为False：

    .. math::
        p_{t+1} = p_{t} - lr \ast v_{t+1}

    其中，:math:`grad` 、:math:`lr` 、:math:`p` 、:math:`v` 和 :math:`u` 分别表示梯度、学习率、参数、矩（Moment）和动量（Momentum）。

    .. note::
        .. include:: mindspore.nn.optim_note_weight_decay.rst

    参数：
        - **params** (Union[list[Parameter], list[dict]]) - 必须是 `Parameter` 组成的列表或字典组成的列表。当列表元素是字典时，字典的键可以是"params"、"lr"、"weight_decay"、"grad_centralization"和"order_params"：

          .. include:: mindspore.nn.optim_group_param.rst
          .. include:: mindspore.nn.optim_group_lr.rst
          .. include:: mindspore.nn.optim_group_dynamic_weight_decay.rst
          .. include:: mindspore.nn.optim_group_gc.rst
          .. include:: mindspore.nn.optim_group_order.rst

        - **learning_rate** (Union[float, int, Tensor, Iterable, LearningRateSchedule]) - 

          .. include:: mindspore.nn.optim_arg_dynamic_lr.rst

        - **momentum** (float) - 浮点数类型的超参，表示移动平均的动量。必须等于或大于0.0。
        - **weight_decay** (Union[float, int, Cell]) - 权重衰减（L2 penalty）。默认值：0.0。

          .. include:: mindspore.nn.optim_arg_dynamic_wd.rst

        .. include:: mindspore.nn.optim_arg_loss_scale.rst

        - **use_nesterov** (bool) - 是否使用Nesterov Accelerated Gradient (NAG)算法更新梯度。默认值：False。

    输入：
        - **gradients** (tuple[Tensor]) - `params` 的梯度，形状（shape）与 `params` 相同。

    输出：
        tuple[bool]，所有元素都为True。

    异常：
        - **TypeError** - `learning_rate` 不是int、float、Tensor、Iterable或LearningRateSchedule。
        - **TypeError** - `parameters` 的元素不是 `Parameter` 或字典。
        - **TypeError** - `loss_scale` 或 `momentum` 不是float。
        - **TypeError** - `weight_decay` 不是float或int。
        - **TypeError** - `use_nesterov` 不是bool。
        - **ValueError** - `loss_scale` 小于或等于0。
        - **ValueError** - `weight_decay` 或 `momentum` 小于0。
