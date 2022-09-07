mindspore.nn.ASGD
==================

.. py:class:: mindspore.nn.ASGD(params, learning_rate=0.1, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0.0)

    随机平均梯度下降（ASGD）算法的实现。
    
    请参阅论文 `Acceleration of stochastic approximation by average <http://dl.acm.org/citation.cfm?id=131098>`_ 。
    
    更新公式如下：
    
    .. math::
        \begin{gather*}
            w_{t} = w_{t-1} * (1 - \lambda * \eta_{t-1}) - \eta_{t-1} * g_{t} \\
            ax_{t} = (w_t - ax_{t-1}) * \mu_{t-1} \\
            \eta_{t} = \frac{1.}{(1 + \lambda * lr * t)^\alpha} \\
            \mu_{t} = \frac{1}{\max(1, t - t0)}
        \end{gather*}
    
    :math:`\lambda` 代表衰减项， :math:`\mu` 和 :math:`\eta` 被跟踪以更新 :math:`ax` 和 :math:`w` ， :math:`t0` 代表开始平均的点， :math:`\α` 代表 :math:`\eta` 更新的系数， :math:`ax` 表示平均参数值， :math:`t` 表示当前步数（step），:math:`g` 表示 `gradients` ， :math:`w` 表示`params` 。

    .. note::
        如果参数未分组，则优化器中的 `weight_decay` 将应用于名称中没有"beta"或"gamma"的参数。用户可以对参数进行分组，以更改权重衰减策略。当参数分组时，每个组都可以设置 `weight_decay` ，如果没有，将应用优化器中的 `weight_decay` 。
        
    参数：
        - **params** (Union[list[Parameter], list[dict]]) - 必须是 `Parameter` 组成的列表或字典组成的列表。当列表元素是字典时，字典的键可以是"params"、"lr"、"weight_decay"、"grad_centralization"和"order_params"：

          .. include:: mindspore.nn.optim_group_param.rst
          .. include:: mindspore.nn.optim_group_lr.rst
          .. include:: mindspore.nn.optim_group_dynamic_weight_decay.rst
          .. include:: mindspore.nn.optim_group_gc.rst
          .. include:: mindspore.nn.optim_group_order.rst

        - **learning_rate** (Union[float, int, Tensor, Iterable, LearningRateSchedule]) -

          .. include:: mindspore.nn.optim_arg_dynamic_lr.rst

        - **lambd** (float) - 衰减项。默认值：1e-4。
        - **alpha** (float) -  :math:`\eta` 更新的系数。默认值：0.75。
        - **t0** (float) - 开始平均的点。默认值：1e6。
        - **weight_decay** (Union[float, int, Cell]) - 权重衰减（L2 penalty）。默认值：0.0。

        .. include:: mindspore.nn.optim_arg_dynamic_wd.rst

    输入：
        - **gradients** (tuple[Tensor]) - `params` 的梯度，shape与 `params` 相同。

    输出：
        Tensor[bool]，值为True。

    异常：
        - **TypeError** - 如果 `learning_rate` 不是int、float、Tensor、Iterable或LearningRateSchedule。
        - **TypeError** - 如果 `parameters` 的元素不是Parameter或字典。
        - **TypeError** - 如果 `lambd` 、 `alpha` 或 `t0` 不是float。
        - **TypeError** - 如果 `weight_decay` 既不是float也不是int。
        - **ValueError** - 如果 `weight_decay` 小于0。
