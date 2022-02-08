mindspore.nn.thor
==================

.. py:class:: mindspore.nn.thor(net, learning_rate, damping, momentum, weight_decay=0.0, loss_scale=1.0, batch_size=32, use_nesterov=False, decay_filter=<function <lambda> at 0x0000029724CFA048>, split_indices=None, enable_clip_grad=False, frequency=100)

    通过二阶算法THOR更新参数。

    基于跟踪的、硬件驱动层定向的自然梯度下降计算（THOR）算法论文地址为：

    `THOR: Trace-based Hardware-driven layer-ORiented Natural Gradient Descent Computation <https://www.aaai.org/AAAI21Papers/AAAI-6611.ChenM.pdf>`_

    更新公式如下：

    .. math::
        \begin{array}{ll} \\
            A_i = a_i{a_i}^T \\
            G_i = D_{s_i}{ D_{s_i}}^T \\
            m_i = \beta * m_i + ({G_i^{(k)}}+\lambda I)^{-1}) g_i ({\overline A_{i-1}^{(k)}}+\lambda I)^{-1} \\
            w_i = w_i - \alpha * m_i \\
        \end{array}

    :math:`D_{s_i}` 表示第i层输出的loss函数的导数。
    :math:`a_{i-1}` 表示第i层的输入，它是上一层的激活。
    :math:`\beta` 表示动量， :math:`I` 代表单位矩阵。
    :math:`\overline A` 表示矩阵A的转置。
    :math:`\lambda` 表示'damping'， :math:`g_i` 表示第i层的梯度。
    :math:`\otimes` 表示克罗内克尔积， :math:`\alpha` 表示学习率。

    .. note::
        在分离参数组时，如果权重衰减为正，则每个组的权重衰减将应用于参数。当不分离参数组时，如果 `weight_decay` 为正数，则API中的 `weight_decay` 将应用于名称中没有'beta'或 'gamma'的参数。

        在分离参数组时，如果要集中梯度，请将grad_centralization设置为True，但梯度集中只能应用于卷积层的参数。
        如果非卷积层的参数设置为True，则会报错。

        为了提高参数组的性能，可以支持参数的自定义顺序。

    **参数：**
        
    - **net** (Cell) - 训练网络。
    - **learning_rate** (Tensor) - 学习率的值。
    - **damping** (Tensor) - 阻尼值。
    - **momentum** (float) - float类型的超参数，表示移动平均的动量。至少为0.0。
    - **weight_decay** (int, float) - 权重衰减（L2 penalty）。必须等于或大于0.0。默认值：0.0。
    - **loss_scale** (float) - loss缩放的值。必须大于0.0。一般情况下，使用默认值。默认值：1.0。
    - **batch_size** (int) - batch的大小。默认值：32。
    - **use_nesterov** (bool) - 启用Nesterov动量。默认值：False。
    - **decay_filter** (function) - 用于确定权重衰减应用于哪些层的函数，只有在weight_decay>0时才有效。默认值：lambda x: x.name not in []。
    - **split_indices** (list) - 按A/G层（A/G含义见上述公式）索引设置allreduce融合策略。仅在分布式计算中有效。ResNet50作为一个样本，A/G的层数分别为54层，当split_indices设置为[26,53]时，表示A/G被分成两组allreduce，一组为0~26层，另一组是27~53层。默认值：None。
    - **enable_clip_grad** (bool) - 是否剪切梯度。默认值：False。
    - **frequency** (int) - A/G和$A^{-1}/G^{-1}$的更新间隔。当频率等于N（N大于1）时，A/G和$A^{-1}/G^{-1}$将每N步更新一次，和其他步骤将使用过时的A/G和$A^{-1}/G^{-1}$更新权重。默认值：100。

    **输入：**

    - **gradients** （tuple[Tensor]） - 训练参数的梯度，矩阵维度与训练参数相同。

    **输出：**
    
    tuple[bool]，所有元素都为True。

    **异常：**
    
    - **TypeError** - `learning_rate` 不是张量。
    - **TypeError** - `loss_scale` 、 `momentum` 或 `frequency` 不是浮点数。
    - **TypeError** - `weight_decay` 既不是浮点数也不是整数。
    - **TypeError** - `use_nesterov` 不是布尔值。
    - **TypeError** - `frequency` 不是整数。
    - **ValueError** - `loss_scale` 小于或等于0。
    - **ValueError** - `weight_decay` 或 `momentum` 小于0。
    - **ValueError** - `frequency` 小于2。

    