mindspore.nn.SGD
================

.. py:class:: mindspore.nn.SGD(params, learning_rate=0.1, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False, loss_scale=1.0)

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

    参数：
        - **params** (Union[list[Parameter], list[dict]]) - 当 `params` 为会更新的 `Parameter` 列表时，`params` 中的元素必须为类 `Parameter`。当 `params` 为 `dict` 列表时，"params"、"lr"、"weight_decay"、"grad_centralization"和"order_params"为可以解析的键。

          .. include:: mindspore.nn.optim_group_param.rst
          .. include:: mindspore.nn.optim_group_lr.rst

          - **weight_decay** - 可选。如果键中存在"weight_decay"，则使用对应的值作为权重衰减值。如果没有，则使用优化器中配置的 `weight_decay` 作为权重衰减值。当前 `weight_decay` 仅支持float类型，不支持动态变化。

          .. include:: mindspore.nn.optim_group_gc.rst
          .. include:: mindspore.nn.optim_group_order.rst

        - **learning_rate** (Union[float, int, Tensor, Iterable, LearningRateSchedule]) - 默认值：0.1。

          .. include:: mindspore.nn.optim_arg_dynamic_lr.rst

        - **momentum** (float) - 浮点动量，必须大于等于0.0。默认值：0.0。
        - **dampening** (float) - 浮点动量阻尼值，必须大于等于0.0。默认值：0.0。
        - **weight_decay** (float) - 权重衰减（L2 penalty），必须大于等于0。默认值：0.0。
        - **nesterov** (bool) - 启用Nesterov动量。如果使用Nesterov，动量必须为正，阻尼必须等于0.0。默认值：False。

        .. include:: mindspore.nn.optim_arg_loss_scale.rst

    输入：
        - **gradients** (tuple[Tensor]) - `params` 的梯度，shape与 `params` 相同。

    输出：
        Tensor[bool]，值为True。

    异常：
        - **ValueError** - 动量、阻尼或重量衰减值小于0.0。
