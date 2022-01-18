mindspore.nn.Lamb
==================

.. py:class:: mindspore.nn.Lamb(*args, **kwargs)

    LAMB（Layer-wise Adaptive Moments optimizer for Batching training，用于批训练的分层自适应矩优化器）算法的实现。

    LAMB是一种采用分层自适应批优化技术的优化算法。详见论文 `LARGE BATCH OPTIMIZATION FOR DEEP LEARNING: TRAINING BERT IN 76 MINUTES <https://arxiv.org/abs/1904.00962>`_。

    LAMB优化器旨在不降低精度的情况下增加训练batch size，支持自适应逐元素更新和精确的分层校正。


    参数更新如下：

    ..  math::
        \begin{gather*}
        m_t = \beta_1 m_{t - 1}+ (1 - \beta_1)g_t\\
        v_t = \beta_2 v_{t - 1}  + (1 - \beta_2)g_t^2\\
        m_t = \frac{m_t}{\beta_1^t}\\
        v_t = \frac{v_t}{\beta_2^t}\\
        r_t = \frac{m_t}{\sqrt{v_t}+\epsilon}\\
        w_t = w_{t-1} -\eta_t \frac{\| w_{t-1} \|}{\| r_t + \lambda w_{t-1} \|} (r_t + \lambda w_{t-1})
        \end{gather*}

    其中， :math:`m` 代表第一个矩向量，:math:`v` 代表第二个矩向量，:math:`\eta` 表示学习率，:math:`\lambda` 表示LAMB权重衰减率。

    .. note::
        .. include:: mindspore.nn.optim_note_weight_decay.rst

        .. include:: mindspore.nn.optim_note_loss_scale.rst

    **参数：**

    - **params** (Union[list[Parameter], list[dict]]): 必须是 `Parameter` 组成的列表或字典组成的列表。当列表元素是字典时，字典的键可以是"params"、"lr"、"weight_decay"、"grad_centralization"和"order_params"：

      .. include:: mindspore.nn.optim_group_param.rst
      .. include:: mindspore.nn.optim_group_lr.rst
      .. include:: mindspore.nn.optim_group_weight_decay.rst
      .. include:: mindspore.nn.optim_group_gc.rst
      .. include:: mindspore.nn.optim_group_order.rst

    - **learning_rate** (Union[float, Tensor, Iterable, LearningRateSchedule]):

      .. include:: mindspore.nn.optim_arg_dynamic_lr.rst

    - **beta1** (float)：第一矩的指数衰减率。参数范围（0.0,1.0）。默认值：0.9。
    - **beta2** (float)：第二矩的指数衰减率。参数范围（0.0,1.0）。默认值：0.999。
    - **eps** (float) - 将添加到分母中，以提高数值稳定性。必须大于0。默认值：1e-6。
    - **weight_decay** (float) - 权重衰减（L2 penalty）。必须大于等于0。默认值：0.0。

    **输入：**

    - **gradients** (tuple[Tensor]) - `params` 的梯度，shape与 `params` 相同。

    **输出：**

    tuple[bool]，所有元素都为True。

    **异常：**

    - **TypeError** - `learning_rate` 不是int、float、Tensor、Iterable或LearningRateSchedule。
    - **TypeError** - `parameters` 的元素不是Parameter或dict。
    - **TypeError** - `beta1`、`beta2` 或 `eps` 不是float。
    - **TypeError** - `weight_decay` 不是float或int。
    - **ValueError** - `eps` 小于等于0。
    - **ValueError** - `beta1`、`beta2` 不在（0.0,1.0）范围内。
    - **ValueError** - `weight_decay` 小于0。
