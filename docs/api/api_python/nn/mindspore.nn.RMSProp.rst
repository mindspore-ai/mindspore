mindspore.nn.RMSProp
======================

.. py:class:: mindspore.nn.RMSProp(params, learning_rate=0.1, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, centered=False, loss_scale=1.0, weight_decay=0.0)

    均方根传播（RMSProp）算法的实现。

    根据RMSProp算法更新 `params`。算法详见 [http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf] 第29页。

    公式如下：

    .. math::
        s_{t+1} = \rho s_{t} + (1 - \rho)(\nabla Q_{i}(w))^2

    .. math::
        m_{t+1} = \beta m_{t} + \frac{\eta} {\sqrt{s_{t+1} + \epsilon}} \nabla Q_{i}(w)

    .. math::
        w = w - m_{t+1}

    第一个方程计算每个权重的平方梯度的移动平均。然后将梯度除以 :math:`\sqrt{ms_{t+1} + \epsilon}`。

    如果centered为True：

    .. math::
        g_{t+1} = \rho g_{t} + (1 - \rho)\nabla Q_{i}(w)

    .. math::
        s_{t+1} = \rho s_{t} + (1 - \rho)(\nabla Q_{i}(w))^2

    .. math::
        m_{t+1} = \beta m_{t} + \frac{\eta} {\sqrt{s_{t+1} - g_{t+1}^2 + \epsilon}} \nabla Q_{i}(w)

    .. math::
        w = w - m_{t+1}

    其中 :math:`w` 代表待更新的网络参数 `params`。
    :math:`g_{t+1}` 是平均梯度。
    :math:`s_{t+1}` 是均方梯度。
    :math:`m_{t+1}` 是moment，`w` 的delta。
    :math:`\rho` 代表 `decay`。:math:`\beta` 是动量项，表示 `momentum`。
    :math:`\epsilon` 是平滑项，可以避免除以零，表示 `epsilon`。
    :math:`\eta` 是学习率，表示 `learning_rate`。 :math:`\nabla Q_{i}(w)` 是梯度，表示 `gradients`。
    :math:`t` 表示当前step。

    .. note::
        .. include:: mindspore.nn.optim_note_weight_decay.rst

    参数：
        - **params** (Union[list[Parameter], list[dict]]) - 必须是 `Parameter` 组成的列表或字典组成的列表。当列表元素是字典时，字典的键可以是"params"、"lr"、"weight_decay"、"grad_centralization"和"order_params"：

          .. include:: mindspore.nn.optim_group_param.rst
          .. include:: mindspore.nn.optim_group_lr.rst
          .. include:: mindspore.nn.optim_group_dynamic_weight_decay.rst
          .. include:: mindspore.nn.optim_group_gc.rst
          .. include:: mindspore.nn.optim_group_order.rst

        - **learning_rate** (Union[float, int, Tensor, Iterable, LearningRateSchedule]) - 默认值：0.1。

          .. include:: mindspore.nn.optim_arg_dynamic_lr.rst

        - **decay** (float) - 衰减率。必须大于等于0。默认值：0.9。
        - **momentum** (float) - Float类型的超参数，表示移动平均的动量（momentum）。必须大于等于0。默认值：0.0。
        - **epsilon** (float) - 将添加到分母中，以提高数值稳定性。取值大于0。默认值：1e-10。
        - **use_locking** (bool) - 是否对参数更新加锁保护。默认值：False。
        - **centered** (bool) - 如果为True，则梯度将通过梯度的估计方差进行归一。默认值：False。

        .. include:: mindspore.nn.optim_arg_loss_scale.rst

        - **weight_decay** (Union[float, int, Cell]) - 权重衰减（L2 penalty）。默认值：0.0。

          .. include:: mindspore.nn.optim_arg_dynamic_wd.rst

    输入：
        - **gradients** （tuple[Tensor]） - `params` 的梯度，shape与 `params` 相同。

    输出：
        Tensor[bool]，值为True。

    异常：
        - **TypeError** - `learning_rate` 不是int、float、Tensor、Iterable或LearningRateSchedule。
        - **TypeError** - `decay` 、 `momentum` 、 `epsilon` 或 `loss_scale` 不是float。
        - **TypeError** - `parameters` 的元素不是Parameter或字典。
        - **TypeError** - `weight_decay` 不是float或int。
        - **TypeError** - `use_locking` 或 `centered` 不是bool。
        - **ValueError** - `epsilon` 小于或等于0。
        - **ValueError** - `decay` 或 `momentum` 小于0。
