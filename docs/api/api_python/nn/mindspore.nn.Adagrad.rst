mindspore.nn.Adagrad
=====================

.. py:class:: mindspore.nn.Adagrad(*args, **kwargs)

    Adagrad算法的实现。

    Adagrad用于在线学习和随机优化。
    请参阅论文 `Efficient Learning using Forward-Backward Splitting <https://proceedings.neurips.cc/paper/2009/file/621bf66ddb7c962aa0d22ac97d69b793-Paper.pdf>`_。

    公式如下：

    .. math::
        \begin{array}{ll} \\
            h_{t+1} = h_{t} + g*g\\
            w_{t+1} = w_{t} - lr*\frac{1}{\sqrt{h_{t+1}}}*g
        \end{array}

    :math:`h` 表示梯度平方的累积和，:math:`g` 表示 `grads` 。
    :math:`lr` 代表 `learning_rate`，:math:`w` 代表 `params` 。

    .. note::
        .. include:: mindspore.nn.optim_note_weight_decay.rst

    **参数：**

    - **params** (Union[list[Parameter], list[dict]]) - 必须是 `Parameter` 组成的列表或字典组成的列表。当列表元素是字典时，字典的键可以是"params"、"lr"、"weight_decay"、"grad_centralization"和"order_params"：

      .. include:: mindspore.nn.optim_group_param.rst
      .. include:: mindspore.nn.optim_group_lr.rst
      .. include:: mindspore.nn.optim_group_weight_decay.rst
      .. include:: mindspore.nn.optim_group_gc.rst
      .. include:: mindspore.nn.optim_group_order.rst

    - **accum** (float) - 累加器 :math:`h` 的初始值，必须大于等于零。默认值：0.1。
    - **learning_rate** (Union[float, Tensor, Iterable, LearningRateSchedule]) - 默认值：0.001。

      .. include:: mindspore.nn.optim_arg_dynamic_lr.rst

    - **update_slots** (bool) - 如果为True，则更新累加器 :math:`h` 。默认值：True。

    .. include:: mindspore.nn.optim_arg_loss_scale.rst

    - **weight_decay** (Union[float, int]) - 要乘以权重的权重衰减值，必须大于等于0.0。默认值：0.0。

    **输入：**

    **grads** (tuple[Tensor]) - 优化器中 `params` 的梯度，形状（shape）与 `params` 相同。

    **输出：**

    Tensor[bool]，值为True。

    **异常：**

    - **TypeError** - `learning_rate` 不是int、float、Tensor、Iterable或 `LearningRateSchedule` 。
    - **TypeError** - `parameters` 的元素是 `Parameter` 或字典。
    - **TypeError** - `accum` 或 `loss_scale` 不是float。
    - **TypeError** - `update_slots` 不是bool。
    - **TypeError** - `weight_decay` 不是float或int。
    - **ValueError** - `loss_scale` 小于或等于0。
    - **ValueError** - `accum` 或 `weight_decay` 小于0。
