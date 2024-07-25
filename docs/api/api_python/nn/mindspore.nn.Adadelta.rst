mindspore.nn.Adadelta
=====================

.. py:class:: mindspore.nn.Adadelta(params, learning_rate=1.0, rho=0.9, epsilon=1e-6, loss_scale=1.0, weight_decay=0.0)

    Adadelta算法的实现。

    Adadelta用于在线学习和随机优化。
    请参阅论文 `ADADELTA: AN ADAPTIVE LEARNING RATE METHOD <https://arxiv.org/pdf/1212.5701.pdf>`_.

    .. math::
        \begin{array}{ll} \\
            accum_{t} = \rho * accum_{t-1} + (1 - \rho) * g_{t}^2 \\
            update_{t} = \sqrt{accum\_update_{t-1} + \epsilon} * \frac{g_{t}}{\sqrt{accum_{t} + \epsilon}} \\
            accum\_update_{t} = \rho * accum\_update_{t-1} + (1 - \rho) * update_{t}^2 \\
            w_{t} = w_{t-1} - \gamma * update_{t}
        \end{array}

    其中 :math:`g` 代表 `grads` ， :math:`\gamma` 代表 `learning_rate` ， :math:`\rho` 代表 `rho` ，
    :math:`\epsilon` 代表 `epsilon` ， :math:`w` 代表 `params` ，
    :math:`accum` 表示累加器， :math:`accum\_update` 表示累加器更新，
    :math:`t` 代表当前step。

    .. note::
        .. include:: mindspore.nn.optim_note_weight_decay.rst

    参数：
        - **params** (Union[list[Parameter], list[dict]]) - 必须是 `Parameter` 组成的列表或字典组成的列表。当列表元素是字典时，字典的键可以是"params"、"lr"、"weight_decay"、"grad_centralization"和"order_params"：

          .. include:: mindspore.nn.optim_group_param.rst
          .. include:: mindspore.nn.optim_group_lr.rst
          .. include:: mindspore.nn.optim_group_weight_decay.rst
          .. include:: mindspore.nn.optim_group_gc.rst
          .. include:: mindspore.nn.optim_group_order.rst

        - **learning_rate** (Union[float, int, Tensor, Iterable, LearningRateSchedule]) - 默认值：1.0。

          .. include:: mindspore.nn.optim_arg_dynamic_lr.rst

        - **rho** (float) - 衰减率，应在 [0.0, 1.0] 范围内。默认值：0.9。
        - **epsilon** (float) - 分母添加项，非负数。默认值：1e-6。

        .. include:: mindspore.nn.optim_arg_loss_scale.rst

        - **weight_decay** (Union[float, int, Cell]) - 权重衰减值，必须大于等于0.0。默认值：0.0。

          - float：固定的权量衰减值。必须等于或大于0。
          - int：固定的权量衰减值。必须等于或大于0。它将会被转换为float类型。
          - Cell：权重衰减此时是动态的。在训练期间，优化器调用该Cell的实例，以获取当前阶段所要使用的权重衰减值。

    输入：
        - **grads** (tuple[Tensor]) - 优化器中 `params` 的梯度，形状（shape）和数据类型与 `params` 相同。数据类型为float16或float32。

    输出：
        Tensor[bool]，值为True。

    异常：
        - **TypeError** - `learning_rate` 不是int、float、Tensor、Iterable或 `LearningRateSchedule` 。
        - **TypeError** - `parameters` 的元素是 `Parameter` 或字典。
        - **TypeError** - `rho` 、 `epsilon` 或 `loss_scale` 不是float。
        - **TypeError** - `weight_decay` 不是float或int。
        - **ValueError** - `rho` 不在范围 [0.0, 1.0] 内。
        - **ValueError** - `loss_scale` 小于或等于0。
        - **ValueError** - `learning_rate` 、 `epsilon` 或 `weight_decay` 小于0。
