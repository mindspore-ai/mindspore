mindspore.nn.Optimizer
======================

.. py:class:: mindspore.nn.Optimizer(learning_rate, parameters, weight_decay=0.0, loss_scale=1.0)

    用于参数更新的优化器基类。不要直接使用这个类，请实例化它的一个子类。

    优化器支持参数分组。当参数分组时，每组参数均可配置不同的学习率（`lr` ）、权重衰减（`weight_decay`）和梯度中心化（`grad_centralization`）策略。

    .. note::
        .. include:: mindspore.nn.optim_note_weight_decay.rst

    **参数：**

    - **learning_rate** (Union[float, int, Tensor, Iterable, LearningRateSchedule]) - 

      .. include:: mindspore.nn.optim_arg_dynamic_lr.rst

    - **parameters** (Union[list[Parameter], list[dict]]) - 必须是 `Parameter` 组成的列表或字典组成的列表。当列表元素是字典时，字典的键可以是"params"、"lr"、"weight_decay"、"grad_centralization"和"order_params"：

      .. include:: mindspore.nn.optim_group_param.rst
      .. include:: mindspore.nn.optim_group_lr.rst
      .. include:: mindspore.nn.optim_group_weight_decay.rst
      .. include:: mindspore.nn.optim_group_gc.rst
      .. include:: mindspore.nn.optim_group_order.rst

    - **weight_decay** (Union[float, int]) - 权重衰减的整数或浮点值。必须等于或大于0。如果 `weight_decay` 是整数，它将被转换为浮点数。默认值：0.0。

    .. include:: mindspore.nn.optim_arg_loss_scale.rst

    **异常：**

    - **TypeError** - `learning_rate` 不是int、float、Tensor、Iterable或LearningRateSchedule。
    - **TypeError** - `parameters` 的元素不是Parameter或字典。
    - **TypeError** - `loss_scale` 不是float。
    - **TypeError** - `weight_decay` 不是float或int。
    - **ValueError** - `loss_scale` 小于或等于0。
    - **ValueError** - `weight_decay` 小于0。
    - **ValueError** - `learning_rate` 是一个Tensor，但是Tensor的维度大于1。

    .. py:method:: broadcast_params(optim_result)

        按参数组的顺序进行参数广播。

        **参数：**

        - **optim_result** (bool) - 参数更新结果。该输入用来保证参数更新完成后才执行参数广播。

        **返回：**

        bool，状态标志。

    .. py:method:: decay_weight(gradients)

        衰减权重。

        一种减少深度学习神经网络模型过拟合的方法。继承  :class:`mindspore.nn.Optimizer` 自定义优化器时，可调用该接口进行权重衰减。

        **参数：**

        - **gradients** (tuple[Tensor]) - 网络参数的梯度，形状（shape）与网络参数相同。

        **返回：**

        tuple[Tensor]，衰减权重后的梯度。

    .. py:method:: flatten_gradients(gradients)

        如果网络参数已经使用了连续内存，则将梯度也按数据类型分组使用连续内存。

        一种网络参数和梯度都使用连续内存的性能优化方法。继承 :class:`mindspore.nn.Optimizer` 自定义优化器时，需调用该接口使能连续内存优化。

        **参数：**

        - **gradients** (tuple[Tensor]) - 网络参数的梯度，形状（shape）与网络参数相同。

        **返回：**

        tuple[Tensor]，如果网络参数使用了连续内存，则返回按数据类型分组使用连续内存后的梯度，否则原样返回输入的梯度。

    .. py:method:: get_lr()

        优化器调用该接口获取当前步骤（step）的学习率。继承 :class:`mindspore.nn.Optimizer` 自定义优化器时，可在参数更新前调用该接口获取学习率。

        **返回：**

        float，当前步骤的学习率。

    .. py:method:: get_lr_parameter(param)

        用于在使用网络参数分组功能，且为不同组别配置不同的学习率时，获取指定参数的学习率。

        **参数：**

        - **param** (Union[Parameter, list[Parameter]]) - `Parameter` 或 `Parameter` 列表。

        **返回：**

        Parameter，单个 `Parameter` 或 `Parameter` 列表。如果使用了动态学习率，返回用于计算学习率的 `LearningRateSchedule` 或 `LearningRateSchedule` 列表。

    .. py:method:: get_weight_decay()

        优化器调用该接口获取当前步骤（step）的weight decay值。继承 :class:`mindspore.nn.Optimizer` 自定义优化器时，可在参数更新前调用该接口获取weight decay值。

        **返回：**

        float，当前步骤的weight decay值。

    .. py:method:: gradients_centralization(gradients)

        梯度中心化。

        一种优化卷积层参数以提高深度学习神经网络模型训练速度的方法。继承 :class:`mindspore.nn.Optimizer` 自定义优化器时，可调用该接口进行梯度中心化。

        **参数：**

        - **gradients** (tuple[Tensor]) - 网络参数的梯度，形状（shape）与网络参数相同。

        **返回：**

        tuple[Tensor]，梯度中心化后的梯度。

    .. py:method:: scale_grad(gradients)

        用于在混合精度场景还原梯度。

        继承 :class:`mindspore.nn.Optimizer` 自定义优化器时，可调用该接口还原梯度。

        **参数：**

        - **gradients** (tuple[Tensor]) - 网络参数的梯度，形状（shape）与网络参数相同。

        **返回：**

        tuple[Tensor]，还原后的梯度。

    .. include:: mindspore.nn.optim_target_unique_for_sparse.rst

    .. include:: mindspore.nn.optim_target_unique_for_sparse.b.rst