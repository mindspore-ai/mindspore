mindspore.nn.probability.bnn_layers.DenseLocalReparam
=====================================================

.. py:class:: mindspore.nn.probability.bnn_layers.DenseLocalReparam(in_channels, out_channels, activation=None, has_bias=True, weight_prior_fn=NormalPrior, weight_posterior_fn=normal_post_fn, bias_prior_fn=NormalPrior, bias_posterior_fn=normal_post_fn)

    具有局部重构化参数的密集变分层。

    更多相关信息，请查阅论文 `变分 Dropout 和局部重参数化技巧 <https://arxiv.org/abs/1506.02557>`_。

    将密集连接层应用于输入。该层将操作实现为：

    .. math::
        \text{outputs} = \text{activation}(\text{inputs} * \text{weight} + \text{bias}),

    此公式中，activation 为激活函数（若 `activation` 参数传入），是与创建层的输入具有相同数据类型的权重矩阵。weight 是从权重的后验分布采样的权重矩阵。bias 是与由层创建的输入具有相同数据类型的偏置向量（仅当 `has_bias` 为 True 时），从 bias 的后验分布中采样。

    参数：
        - **in_channels** (int) - 输入通道的数量。
        - **out_channels** (int) - 输出通道的数量。
        - **activation** (str, Cell) - 应用于输出层的正则化函数。激活的类型可以是 string（例如'relu'）或 Cell（例如nn.ReLU()）。注意，如果激活的类型是 Cell，则必须事先实例化。默认值：None。
        - **has_bias** (bool) - 指定层是否使用偏置向量。默认值：True。 
        - **weight_prior_fn** (Cell) - 它必须返回一个 mindspore 分布实例。默认值：NormalPrior。（创建标准正态分布的一个实例）。当前版本仅支持正态分布。
        - **weight_posterior_fn** (function) - 采样权重的后验分布。它必须是一个函数句柄，它返回一个 mindspore 分布实例。默认值：normal_post_fn。当前版本仅支持正态分布。
        - **bias_prior_fn** (Cell) - 偏置向量的先验分布。它必须返回一个 mindspore 分布实例。默认值：NormalPrior（创建标准正态分布的实例）。当前版本仅支持正态分布。
        - **bias_posterior_fn** (function) - 采样偏差向量的后验分布。它必须是一个函数句柄，它返回一个 mindspore 分布实例。默认值：normal_post_fn。当前版本仅支持正态分布。

    输入：
        - **input** (Tensor) - `input` 的 shape 是 :math:`(N, in\_channels)`。

    输出：
        Tensor，`output` 的 shape 是 :math:`(N, out\_channels)`。