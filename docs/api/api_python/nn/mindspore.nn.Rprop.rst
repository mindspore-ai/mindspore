mindspore.nn.Rprop
===================

.. py:class:: mindspore.nn.Rprop(params, learning_rate=0.1, etas=(0.5, 1.2), step_sizes=(1e-6, 50.), weight_decay=0.)

    弹性反向传播（Rprop）算法的实现。

    请参阅论文 `A Direct Adaptive Method for Faster Backpropagation Learning: The RPROP Algorithm. <https://ieeexplore.ieee.org/document/298623>`_ 。

    更新公式如下：

    .. math::
        \begin{gather*}
            &\hspace{-10mm}  \textbf{if} \:   g_{t-1} g_t  > 0                                     \\
            &\hspace{25mm}  \Delta_t \leftarrow \mathrm{min}(\Delta_{t-1} \eta_{+}, \Delta_{max})  \\
            &\hspace{0mm}  \textbf{else if}  \:  g_{t-1} g_t < 0                                   \\
            &\hspace{25mm}  \Delta_t \leftarrow \mathrm{max}(\Delta_{t-1} \eta_{-}, \Delta_{min})  \\
            &\hspace{-25mm}  \textbf{else}  \:                                                     \\
            &\hspace{-5mm}  \Delta_t \leftarrow \Delta_{t-1}                                       \\
            &\hspace{15mm} w_{t} \leftarrow w_{t-1}- \Delta_{t} \mathrm{sign}(g_t)                 \\
        \end{gather*}

    :math:`\Delta_{min/max}` 表示最小或者最大步长， :math:`\eta_{+/-}` 表示加速和减速因子， :math:`g` 表示 `gradients` ， :math:`w` 表示 `params` 。

    .. note::
        .. include:: mindspore.nn.optim_note_weight_decay.rst

    参数：
        - **params** (Union[list[Parameter], list[dict]]) - 必须是 `Parameter` 组成的列表或字典组成的列表。当列表元素是字典时，字典的键可以是"params"、"lr"、"weight_decay"、"grad_centralization"和"order_params"：

          .. include:: mindspore.nn.optim_group_param.rst
          .. include:: mindspore.nn.optim_group_lr.rst
          .. include:: mindspore.nn.optim_group_dynamic_weight_decay.rst
          .. include:: mindspore.nn.optim_group_gc.rst
          .. include:: mindspore.nn.optim_group_order.rst

        - **learning_rate** (Union[float, int, Tensor, Iterable, LearningRateSchedule]) - 学习率。默认值：0.1。

          .. include:: mindspore.nn.optim_arg_dynamic_lr.rst

        - **etas** (tuple[float, float]) - 乘法的增加或减少的因子（etaminus, etaplus）。默认值：(0.5, 1.2)。
        - **step_sizes** (tuple[float, float]) - 允许的最小和最大步长（min_step_sizes, max_step_size）。默认值：(1e-6, 50.)。
        - **weight_decay** (Union[float, int, Cell]) - 权重衰减（L2 penalty）。默认值：0.0。

          .. include:: mindspore.nn.optim_arg_dynamic_wd.rst

    输入：
        - **gradients** (tuple[Tensor]) - `params` 的梯度，shape与 `params` 相同。

    输出：
        Tensor[bool]，值为True。

    异常：
        - **TypeError** - 如果 `learning_rate` 不是int、float、Tensor、Iterable或LearningRateSchedule。
        - **TypeError** - 如果 `parameters` 的元素不是Parameter或字典。
        - **TypeError** - 如果 `step_sizes` 或 `etas` 不是tuple。
        - **ValueError** - 如果最大步长小于最小步长。
        - **ValueError** - 如果 `step_sizes` 或 `etas` 的长度不等于2。
        - **TypeError** - 如果 `etas` 或 `step_sizes` 中的元素不是float。
        - **ValueError** - 如果 `etaminus` 不在（0,1）的范围内，或者 `etaplus` 不大于1。
        - **TypeError** - 如果 `weight_decay` 既不是float也不是int。
        - **ValueError** - 如果 `weight_decay` 小于0。
