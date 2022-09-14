mindspore.nn.ProximalAdagrad
==============================

.. py:class:: mindspore.nn.ProximalAdagrad(params, accum=0.1, learning_rate=0.001, l1=0.0, l2=0.0, use_locking=False, loss_scale=1.0, weight_decay=0.0)

    ProximalAdagrad算法的实现。

    ProximalAdagrad用于在线学习和随机优化。
    请参阅论文 `Efficient Learning using Forward-Backward Splitting <http://papers.nips.cc//paper/3793-efficient-learning-using-forward-backward-splitting.pdf>`_。

    .. math::
        accum_{t+1} = accum_{t} + g * g

    .. math::
        \text{prox_v} = w_{t} - \gamma * g * \frac{1}{\sqrt{accum_{t+1}}}

    .. math::
        w_{t+1} = \frac{sign(\text{prox_v})}{1 + \gamma * l2} * \max(\left| \text{prox_v} \right| - \gamma * l1, 0)

    其中， :math:`g` 、 :math:`\gamma` 、 :math:`w` 、 :math:`accum` 和 :math:`t` 分别表示 `grads` 、 `learning_rate` 、 `params` 、累加器和当前step。

    .. note::
        .. include:: mindspore.nn.optim_note_sparse.rst

        .. include:: mindspore.nn.optim_note_weight_decay.rst

    参数：
        - **params** (Union[list[Parameter], list[dict]]) - 必须是 `Parameter` 组成的列表或字典组成的列表。当列表元素是字典时，字典的键可以是"params"、"lr"、"weight_decay"、"grad_centralization"和"order_params"：

          .. include:: mindspore.nn.optim_group_param.rst
          .. include:: mindspore.nn.optim_group_lr.rst
          .. include:: mindspore.nn.optim_group_dynamic_weight_decay.rst
          .. include:: mindspore.nn.optim_group_gc.rst
          .. include:: mindspore.nn.optim_group_order.rst

        - **accum** (float) - 累加器 `accum` 的初始值，起始值必须为零或正值。默认值：0.1。

        - **learning_rate** (Union[float, int, Tensor, Iterable, LearningRateSchedule]) - 默认值：1e-3。

          .. include:: mindspore.nn.optim_arg_dynamic_lr.rst

        - **l1** (float) - l1正则化强度，必须大于或等于零。默认值：0.0。
        - **l2** (float) - l2正则化强度，必须大于或等于零。默认值：0.0。
        - **use_locking** (bool) - 如果为True，则更新操作使用锁保护。默认值：False。

        .. include:: mindspore.nn.optim_arg_loss_scale.rst

        - **weight_decay** (Union[float, int, Cell]) - 权重衰减（L2 penalty）。默认值：0.0。

          .. include:: mindspore.nn.optim_arg_dynamic_wd.rst

    输入：
        - **grads** (tuple[Tensor]) - 优化器中 `params` 的梯度，shape与优化器中的 `params` 相同。

    输出：
        Tensor[bool]，值为True。

    异常：
        - **TypeError** - `learning_rate` 不是int、float、Tensor、Iterable或LearningRateSchedule。
        - **TypeError** - `parameters` 的元素不是Parameter或字典。
        - **TypeError** - `accum`、`l1`、`l2` 或 `loss_scale` 不是float。
        - **TypeError** - `weight_decay` 不是float或int。
        - **ValueError** - `loss_scale` 小于或等于0。
        - **ValueError** - `accum`、`l1`、`l2` 或 `weight_decay` 小于0。
