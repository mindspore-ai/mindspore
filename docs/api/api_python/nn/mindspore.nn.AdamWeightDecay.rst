mindspore.nn.AdamWeightDecay
===============================

.. py:class:: mindspore.nn.AdamWeightDecay(params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0)

    权重衰减Adam算法的实现。

    .. math::
        \begin{array}{l}
            &\newline
            &\hline \\
            &\textbf{Parameters}: \: 1^{\text {st }}\text {moment vector} \: m , \: 2^{\text {nd}} \:
             \text{moment vector} \: v , \\
            &\: gradients \: g, \: \text{learning rate} \: \gamma,
             \text {exponential decay rates for the moment estimates} \: \beta_{1} \: \beta_{2} , \\
            &\:\text {parameter vector} \: w_{0}, \:\text{timestep} \: t, \: \text{weight decay} \: \lambda \\
            &\textbf{Init}:  m_{0} \leftarrow 0, \: v_{0} \leftarrow 0, \: t \leftarrow 0, \:
             \text{init parameter vector} \: w_{0} \\[-1.ex]
            &\newline
            &\hline \\
            &\textbf{repeat} \\
            &\hspace{5mm} t \leftarrow t+1 \\
            &\hspace{5mm}\boldsymbol{g}_{t} \leftarrow \nabla f_{t}\left(\boldsymbol{w}_{t-1}\right) \\
            &\hspace{5mm}\boldsymbol{m}_{t} \leftarrow \beta_{1} \boldsymbol{m}_{t-1}+\left(1-\beta_{1}\right)
             \boldsymbol{g}_{t} \\
            &\hspace{5mm}\boldsymbol{v}_{t} \leftarrow \beta_{2} \boldsymbol{v}_{t-1}+\left(1-\beta_{2}\right)
             \boldsymbol{g}_{t}^{2} \\
            &\hspace{5mm}\boldsymbol{w}_{t} \leftarrow \boldsymbol{w}_{t-1}-\left(\gamma \hat{\boldsymbol{m}}_{t}
             /\left(\sqrt{\hat{\boldsymbol{v}}_{t}}+\epsilon\right)+\lambda \boldsymbol{w}_{t-1}\right) \\
            &\textbf{until}\text { stopping criterion is met } \\[-1.ex]
            &\newline
            &\hline \\[-1.ex]
            &\textbf{return} \: \boldsymbol{w}_{t} \\[-1.ex]
            &\newline
            &\hline \\[-1.ex]
        \end{array}

    :math:`m` 代表第一个动量矩阵 `moment1` ，:math:`v` 代表第二个动量矩阵 `moment2` ，:math:`g` 代表 `gradients` ，:math:`\gamma` 代表 `learning_rate` ，:math:`\beta_1, \beta_2` 代表 `beta1` 和 `beta2` ， :math:`t` 代表当前step，:math:`w` 代表 `params` ，:math:`\lambda` 代表 `weight_decay` 。

    .. note::
        .. include:: mindspore.nn.optim_note_loss_scale.rst
        .. include:: mindspore.nn.optim_note_weight_decay.rst

    参数：
        - **params** (Union[list[Parameter], list[dict]]) - 必须是 `Parameter` 组成的列表或字典组成的列表。当列表元素是字典时，字典的键可以是"params"、"lr"、"weight_decay"、和"order_params"：

          .. include:: mindspore.nn.optim_group_param.rst

          .. include:: mindspore.nn.optim_group_lr.rst

          .. include:: mindspore.nn.optim_group_dynamic_weight_decay.rst

          .. include:: mindspore.nn.optim_group_order.rst

        - **learning_rate** (Union[float, int, Tensor, Iterable, LearningRateSchedule]) - 默认值：1e-3。

          .. include:: mindspore.nn.optim_arg_dynamic_lr.rst

        - **beta1** (float) - `moment1` 的指数衰减率。参数范围（0.0,1.0）。默认值：0.9。
        - **beta2** (float) - `moment2` 的指数衰减率。参数范围（0.0,1.0）。默认值：0.999。
        - **eps** (float) - 将添加到分母中，以提高数值稳定性。必须大于0。默认值：1e-6。
        - **weight_decay** (Union[float, int, Cell]) - 权重衰减（L2 penalty）。默认值：0.0。

          .. include:: mindspore.nn.optim_arg_dynamic_wd.rst

    输入：
        - **gradients** (tuple[Tensor]) - `params` 的梯度，shape与 `params` 相同。

    输出：
        tuple[bool]，所有元素都为True。

    异常：
        - **TypeError** - `learning_rate` 不是int、float、Tensor、Iterable或LearningRateSchedule。
        - **TypeError** - `parameters` 的元素不是Parameter或字典。
        - **TypeError** - `beta1` 、 `beta2` 或 `eps` 不是float。
        - **TypeError** - `weight_decay` 不是float或int。
        - **ValueError** - `eps` 小于等于0。
        - **ValueError** - `beta1` 、 `beta2` 不在（0.0,1.0）范围内。
        - **ValueError** - `weight_decay` 小于0。
