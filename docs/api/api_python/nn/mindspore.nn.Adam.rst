mindspore.nn.Adam
==================

.. py:class:: mindspore.nn.Adam(params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, use_locking=False, use_nesterov=False, weight_decay=0.0, loss_scale=1.0, use_amsgrad=False)

    Adaptive Moment Estimation (Adam)算法的实现。

    Adam optimizer可以使用梯度的first-order moment estimation和second-order moment estimation，动态地调整每一个参数的学习率。

    请参阅论文 `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_。

    公式如下：

    .. math::
        \begin{array}{l}
            &\newline
            &\hline \\
            &\textbf{Parameters}: \: 1^{\text {st }}\text {moment vector} \: m , \: 2^{\text {nd}} \:
             \text{moment vector} \: v , \\
            &\:\text{gradients } g, \: \text{learning rate} \: \gamma, \text
             { exponential decay rates for the moment estimates} \: \beta_{1} \: \beta_{2} , \\
            &\:\text {parameter vector} \: w_{0}, \:\text{timestep} \: t , \text{ weight decay } \lambda \\
            &\textbf{Init}: m_{0} \leftarrow 0, \: v_{0} \leftarrow 0, \: t \leftarrow 0, \:
             \text{init parameter vector} \: w_{0} \\[-1.ex]
            &\newline
            &\hline \\
            &\textbf{while} \: w_{t} \: \text{not converged} \: \textbf{do} \\
            &\hspace{5mm}\boldsymbol{g}_{t} \leftarrow \nabla_{w} \boldsymbol{f}_{t}\left(\boldsymbol{w}_{t-1}\right) \\
            &\hspace{5mm}\textbf {if } \lambda \neq 0 \\
            &\hspace{10mm}\boldsymbol{g}_{t} \leftarrow \boldsymbol{g}_{t}+\lambda \boldsymbol{w}_{t-1} \\
            &\hspace{5mm}\boldsymbol{m}_{t} \leftarrow \beta_{1} \boldsymbol{m}_{t-1}+\left(1-\beta_{1}\right)
             \boldsymbol{g}_{t} \\
            &\hspace{5mm}\boldsymbol{v}_{t} \leftarrow \beta_{2} \boldsymbol{v}_{t-1}+\left(1-\beta_{2}\right)
             \boldsymbol{g}_{t}^{2} \\
            &\hspace{5mm}\hat{\boldsymbol{m}}_{t} \leftarrow \boldsymbol{m}_{t} /\left(1-\beta_{1}^{t}\right) \\
            &\hspace{5mm}\hat{\boldsymbol{v}}_{t} \leftarrow \boldsymbol{v}_{t} /\left(1-\beta_{2}^{t}\right) \\
            &\hspace{5mm}\boldsymbol{w}_{t} \leftarrow \boldsymbol{w}_{t-1}-\gamma \hat{\boldsymbol{m}}_{t}
             /(\sqrt{\hat{\boldsymbol{v}}_{t}}+\epsilon) \\
            &\textbf{end while} \\[-1.ex]
            &\newline
            &\hline \\[-1.ex]
            &\textbf{return} \:  \boldsymbol{w}_{t} \\[-1.ex]
            &\newline
            &\hline \\[-1.ex]
        \end{array}

    :math:`m` 代表第一个动量矩阵，:math:`v` 代表第二个动量矩阵，:math:`g` 代表梯度 `gradients` ，:math:`\gamma` 代表学习率 `learning_rate` ，:math:`\beta_1, \beta_2` 代表衰减速率 `beta1` 和 `beta2` ，:math:`t` 代表当前step，:math:`beta_1^t` 和 :math:`beta_2^t` 代表 `beta1` 和 `beta2` 的t次方 ， :math:`w` 代表 `params` ， :math:`\epsilon` 代表 `eps` 。

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

        - **learning_rate** (Union[float, int, Tensor, Iterable, LearningRateSchedule]) - 默认值：1e-3。

          .. include:: mindspore.nn.optim_arg_dynamic_lr.rst

        - **beta1** (float) - 第一个动量矩阵的指数衰减率。参数范围（0.0,1.0）。默认值：0.9。
        - **beta2** (float) - 第二个动量矩阵的指数衰减率。参数范围（0.0,1.0）。默认值：0.999。
        - **eps** (float) - 加在分母上的值，以确保数值稳定。必须大于0。默认值：1e-8。
        - **use_locking** (bool) - 是否对参数更新加锁保护。如果为True，则 `w` 、`m` 和 `v` 的tensor更新将受到锁的保护。如果为False，则结果不可预测。默认值：False。
        - **use_nesterov** (bool) - 是否使用Nesterov Accelerated Gradient (NAG)算法更新梯度。如果为True，使用NAG更新梯度。如果为False，则在不使用NAG的情况下更新梯度。默认值：False。
        - **use_amsgrad** (bool) - 是否使用Amsgrad算法更新梯度。如果为True，使用Amsgrad更新梯度。如果为False，则在不使用Amsgrad的情况下更新梯度。默认值：False。
        - **weight_decay** (Union[float, int, Cell]) - 权重衰减（L2 penalty）。默认值：0.0。

          .. include:: mindspore.nn.optim_arg_dynamic_wd.rst

        .. include:: mindspore.nn.optim_arg_loss_scale.rst

    输入：
        - **gradients** (tuple[Tensor]) - `params` 的梯度，形状（shape）与 `params` 相同。

    输出：
        Tensor[bool]，值为True。

    异常：
        - **TypeError** - `learning_rate` 不是int、float、Tensor、iterable或LearningRateSchedule。
        - **TypeError** - `parameters` 的元素不是Parameter或字典。
        - **TypeError** - `beta1` 、 `beta2` 、 `eps` 或 `loss_scale` 不是float。
        - **TypeError** - `weight_decay` 不是float或int。
        - **TypeError** - `use_locking` 、 `use_nesterov` 或 `use_amsgrad` 不是bool。
        - **ValueError** - `loss_scale` 或 `eps` 小于或等于0。
        - **ValueError** - `beta1` 、`beta2` 不在（0.0,1.0）范围内。
        - **ValueError** - `weight_decay` 小于0。
