mindspore.nn.thor
==================

.. py:function:: mindspore.nn.thor(net, learning_rate, damping, momentum, weight_decay=0.0, loss_scale=1.0, batch_size=32, use_nesterov=False, decay_filter=lambda x: x.name not in [], split_indices=None, enable_clip_grad=False, frequency=100)

    通过二阶算法THOR更新参数。

    更新公式如下：

    .. math::
        \begin{array}{ll}
          & \textbf{Parameter:} \: \text{the learning rate } \gamma\text{, the damping parameter }\lambda \\
          & \textbf{Init:} \: \lambda \leftarrow 0 \\
          & A_{i-1}=\mathbb{E}\left[a_{i-1} a_{i-1}^{T}\right] \\
          & G_{i}=\mathbb{E}\left[D_{s_i} D_{s_i}^{T}\right] \\
          & w_{i}^{(k+1)} \leftarrow w_{i}^{(k)}-\gamma\left(\left(A_{i-1}^{(k)}+\lambda I\right)^{-1}
            \otimes\left(G_{i}^{(k)}+\lambda I\right)^{-1}\right) \nabla_{w_{i}} J^{(k)}
        \end{array}

    :math:`a_{i-1}` 表示第i层的输入，它是上一层的激活。
    :math:`D_{s_i}` 表示第i层输出的loss函数的导数。
    :math:`I` 代表单位矩阵。
    :math:`\lambda` 表示 :math:`damping` 参数， :math:`g_i` 表示第i层的梯度。
    :math:`\otimes` 表示克罗内克尔积， :math:`\gamma` 表示学习率。

    .. note::
        在分离参数组时，每个组的 `weight_decay` 将应用于对应参数。当不分离参数组时，优化器中的 `weight_decay` 将应用于名称中没有'beta'或 'gamma'的参数。

        在分离参数组时，如果要集中梯度，请将grad_centralization设置为True，但集中梯度只能应用于卷积层的参数。
        如果非卷积层的参数设置为True，则会报错。

        为了提高参数组的性能，可以支持自定义参数的顺序。

    参数：
        - **net** (Cell) - 训练网络。
        - **learning_rate** (Tensor) - 学习率的值。
        - **damping** (Tensor) - 阻尼值。
        - **momentum** (float) - float类型的超参数，表示移动平均的动量。至少为0.0。
        - **weight_decay** (int, float) - 权重衰减（L2 penalty）。必须等于或大于0.0。默认值：0.0。
        - **loss_scale** (float) - loss损失缩放系数。必须大于0.0。一般情况下，使用默认值。默认值：1.0。
        - **batch_size** (int) - batch的大小。默认值：32。
        - **use_nesterov** (bool) - 启用Nesterov动量。默认值：False。
        - **decay_filter** (function) - 用于确定权重衰减应用于哪些层的函数，只有在weight_decay>0时才有效。默认值：lambda x: x.name not in []。
        - **split_indices** (list) - 按A/G层（A/G含义见上述公式）索引设置allreduce融合策略。仅在分布式计算中有效。ResNet50作为一个样本，A/G的层数分别为54层，当split_indices设置为[26,53]时，表示A/G被分成两组allreduce，一组为0~26层，另一组是27~53层。默认值：None。
        - **enable_clip_grad** (bool) - 是否剪切梯度。默认值：False。
        - **frequency** (int) - A/G和 :math:`A^{-1}/G^{-1}` 的更新间隔。当frequency等于N(N必须大于1)，每隔frequency个step，A/G和 :math:`A^{-1}/G^{-1}` 将更新一次。其他step将使用之前的A/G和 :math:`A^{-1}/G^{-1}` 来更新权重。默认值：100。

    输入：
        - **gradients** （tuple[Tensor]） - 训练参数的梯度，矩阵维度与训练参数相同。

    输出：
        tuple[bool]，所有元素都为True。

    异常：
        - **TypeError** - `learning_rate` 不是张量。
        - **TypeError** - `loss_scale` 、 `momentum` 或 `frequency` 不是浮点数。
        - **TypeError** - `weight_decay` 既不是浮点数也不是整数。
        - **TypeError** - `use_nesterov` 不是布尔值。
        - **TypeError** - `frequency` 不是整数。
        - **ValueError** - `loss_scale` 小于或等于0。
        - **ValueError** - `weight_decay` 或 `momentum` 小于0。
        - **ValueError** - `frequency` 小于2。

    样例：

    .. note::
        运行以下样例之前，需自定义网络Net和数据集准备函数create_dataset。详见 `网络构建 <https://www.mindspore.cn/tutorials/zh-CN/master/beginner/model.html>`_ 和 `数据集 Dataset <https://www.mindspore.cn/tutorials/zh-CN/master/beginner/dataset.html>`_ 。