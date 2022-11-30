mindspore.nn.Lamb
==================

.. py:class:: mindspore.nn.Lamb(params, learning_rate, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0)

    LAMB（Layer-wise Adaptive Moments optimizer for Batching training，用于批训练的分层自适应矩优化器）算法的实现。

    LAMB是一种采用分层自适应批优化技术的优化算法。详见论文 `LARGE BATCH OPTIMIZATION FOR DEEP LEARNING: TRAINING BERT IN 76 MINUTES <https://arxiv.org/abs/1904.00962>`_。

    LAMB优化器旨在不降低精度的情况下增加训练batch size，支持自适应逐元素更新和精确的分层校正。

    参数更新如下：

    ..  math::
        \begin{array}{l}
            &\newline
            &\hline \\
            &\textbf{Parameters}:   \: 1^{\text {st }}\text {moment vector} \: m , \: 2^{\text {nd}} \:
             \text{moment vector} \: v , \\
            &\hspace{5mm}\text{learning rate }  \left\{ \gamma_{t}\right\}_{t=1}^{T} , \: \text
             {exponential decay rates for the moment estimates} \: \beta_{1} \: \beta_{2} , \\
            &\hspace{5mm}\text{scaling function } \phi \\
            &\textbf{Init}: \boldsymbol{m}_{0} \leftarrow 0, \: \boldsymbol{v}_{0} \leftarrow 0 \\[-1.ex]
            &\newline
            &\hline \\
            &\textbf{for} \text { t=1  to  T } \textbf{do} \\
            &\hspace{5mm}\text{Draw b samples } S_{t} \text{ from } \mathbb{P} \text{ . } \\
            &\hspace{5mm}\text{Compute } g_{t}=\frac{1}{\left|\mathcal{S}_{t}\right|} \sum_{s_{t} \in \mathcal{S}_{t}}
             \nabla \ell\left(x_{t}, s_{t}\right) . \\
            &\hspace{5mm}\boldsymbol{m}_{t} \leftarrow \beta_{1} \boldsymbol{m}_{t-1}+\left(1-\beta_{1}\right)
             \boldsymbol{g}_{t} \\
            &\hspace{5mm}\boldsymbol{v}_{t} \leftarrow \beta_{2} \boldsymbol{v}_{t-1}+\left(1-\beta_{2}\right)
             \boldsymbol{g}_{t}^{2} \\
            &\hspace{5mm}\hat{\boldsymbol{m}}_{t} \leftarrow \boldsymbol{m}_{t} /\left(1-\beta_{1}^{t}\right) \\
            &\hspace{5mm}\hat{\boldsymbol{v}}_{t} \leftarrow \boldsymbol{v}_{t} /\left(1-\beta_{2}^{t}\right) \\
            &\hspace{5mm}\text{Compute ratio } \boldsymbol{r}_{t}=\hat{\boldsymbol{m}}_{t}
             /(\sqrt{\hat{\boldsymbol{v}}_{t}}+\epsilon) \\
            &\hspace{5mm}\boldsymbol{w}_{t+1}^{(i)}=\boldsymbol{w}_{t}^{(i)}- \gamma_{t}
             \frac{\boldsymbol{\phi}\left(\left\|\boldsymbol{w}_{t}^{(i)}\right\|\right)}
             {\left\|\boldsymbol{r}_{t}^{(i)}+\lambda \boldsymbol{w}_{t}^{(i)}\right\|}\left(\boldsymbol{r}_{t}^{(i)}+
             \lambda \boldsymbol{w}_{t}^{(i)}\right) \\
            &\textbf{end for} \\[-1.ex]
            &\newline
            &\hline \\[-1.ex]
            &\textbf{return} \: \boldsymbol{w}_{t+1}\\[-1.ex]
            &\newline
            &\hline \\[-1.ex]
        \end{array}

    其中， :math:`m` 代表第一个动量矩阵 `moment1` ，:math:`v` 代表第二个动量矩阵 `moment2` ，:math:`g` 代表梯度 `gradients` ，:math:`\gamma` 代表学习率 `learning_rate`，:math:`\beta_1, \beta_2` 代表衰减速率 `beta1` 和 `beta2` ，:math:`t` 代表当前step，:math:`beta_1^t` 和 :math:`beta_2^t` 代表 `beta1` 和 `beta2` 的t次方 ， :math:`w` 代表 `params` ， :math:`\epsilon` 代表 `eps`， :math:`\lambda` 表示LAMB权重衰减率。

    .. note::
        .. include:: mindspore.nn.optim_note_weight_decay.rst

        .. include:: mindspore.nn.optim_note_loss_scale.rst

    参数：
        - **params** (Union[list[Parameter], list[dict]]) - 必须是 `Parameter` 组成的列表或字典组成的列表。当列表元素是字典时，字典的键可以是"params"、"lr"、"weight_decay"、"grad_centralization"和"order_params"：

          .. include:: mindspore.nn.optim_group_param.rst
          .. include:: mindspore.nn.optim_group_lr.rst
          .. include:: mindspore.nn.optim_group_dynamic_weight_decay.rst
          .. include:: mindspore.nn.optim_group_gc.rst
          .. include:: mindspore.nn.optim_group_order.rst

        - **learning_rate** (Union[float, int, Tensor, Iterable, LearningRateSchedule]) - 

          .. include:: mindspore.nn.optim_arg_dynamic_lr.rst

        - **beta1** (float) - 第一矩的指数衰减率。参数范围（0.0,1.0）。默认值：0.9。
        - **beta2** (float) - 第二矩的指数衰减率。参数范围（0.0,1.0）。默认值：0.999。
        - **eps** (float) - 将添加到分母中，以提高数值稳定性。必须大于0。默认值：1e-6。
        - **weight_decay** (Union[float, int, Cell]) - 权重衰减（L2 penalty）。默认值：0.0。

          .. include:: mindspore.nn.optim_arg_dynamic_wd.rst

    输入：
        - **gradients** (tuple[Tensor]) - `params` 的梯度，shape与 `params` 相同。

    输出：
        tuple[bool]，所有元素都为True。

    异常：
        - **TypeError** - `learning_rate` 不是int、float、Tensor、Iterable或LearningRateSchedule。
        - **TypeError** - `parameters` 的元素不是Parameter或dict。
        - **TypeError** - `beta1`、`beta2` 或 `eps` 不是float。
        - **TypeError** - `weight_decay` 不是float或int。
        - **ValueError** - `eps` 小于等于0。
        - **ValueError** - `beta1`、`beta2` 不在（0.0,1.0）范围内。
        - **ValueError** - `weight_decay` 小于0。
