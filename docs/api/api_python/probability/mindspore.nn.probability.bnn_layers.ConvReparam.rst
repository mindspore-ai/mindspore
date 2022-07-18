mindspore.nn.probability.bnn_layers.ConvReparam
===============================================

.. py:class:: mindspore.nn.probability.bnn_layers.ConvReparam(in_channels, out_channels, kernel_size, stride=1, pad_mode="same", padding=0, dilation=1, group=1, has_bias=False, weight_prior_fn=NormalPrior, weight_posterior_fn=normal_post_fn, bias_prior_fn=NormalPrior, bias_posterior_fn=normal_post_fn)

    具有重构化参数的卷积变分层。
    更多相关信息可以查阅 `自动编码变分贝叶斯 <https://arxiv.org/abs/1312.6114>`_ 相关论文。

    
    参数：
        - **in_channels** (int) - 输入的channels :math:`C_{in}`。
        - **out_channels** (int) - 输出的channels :math:`C_{out}`。
        - **kernel_size** (Union[int, tuple[int]]) - 数据类型是 int 或2个 int 的元组。内核大小指定二维卷积窗口的高度和宽度。若为一个整数则高度和宽度均为该值，若为元组则两个值分别为高度和宽度。
        - **stride** (Union[int, tuple[int]]) - 内核移动的距离，若是一个整数表示，则移动的高度和宽度都是步幅，或者两个整数的元组分别表示移动的高度和宽度。默认值：1。
        - **pad_mode** (str) - 指定填充模式。可选值是"same"、"valid"和"pad"。默认值："same"。

          - same：采用补全方式。输出高度和宽度将与输入相同。将在水平和垂直方向上计算填充的总数，并尽可能均匀地分布在顶部和底部、左侧和右侧。否则，最后的额外填充将从底部和右侧完成。如果设置了此模式，则 `padding` 必须为0。

          - valid：采用丢弃的方式。输出的可能最大高度和宽度将不带 padding 返回。多余的像素将被丢弃。如果设置了此模式，则 `padding` 必须为0。

          - pad：输入两侧的隐式 padding。 `padding` 的值将被填充到输入 Tensor 边界。 `padding` 必须大于或等于0。

        - **padding** (Union[int, tuple[int]]) - 输入两侧的隐式 padding 。默认值：0。
        - **dilation** (Union[int, tuple[int]]) - 数据类型是 int 或2个 int 的元组。该参数指定空洞卷积的空洞率。如果设置为k>1，将有k−1每个采样位置跳过的像素。它的值必须大于或等于1，并受输入的高度和宽度限制。默认值：1。
        - **group** (int) - 将过滤器拆分为组，`in_channels` 和 `out_channels` 必须能被组数整除。默认值：1。
        - **has_bias** (bool) - 指定层是否使用偏置向量。默认值：false。
        - **weight_prior_fn** (Cell) - 权重的先验分布。它必须返回一个 MindSpore 分布实例。默认值：NormalPrior。（创建标准正态分布的一个实例）。当前版本仅支持正态分布。
        - **weight_posterior_fn** (function) - 采样权重的后验分布。它必须是一个函数句柄，它返回一个 MindSpore 分布实例。默认值：normal_post_fn。当前版本仅支持正态分布。
        - **bias_prior_fn** (Cell) - 偏置向量的先验分布。它必须返回一个 MindSpore 分布实例。默认值：NormalPrior（创建标准正态分布的实例）。当前版本仅支持正态分布。
        - **bias_posterior_fn** (function) - 采样偏差向量的后验分布。它必须是一个函数句柄，它返回一个 MindSpore 分布实例。默认值：normal_post_fn。当前版本仅支持正态分布。
    
    
    输入：
        - **input** (Tensor) - 输入 Tensor 的 shape 为 :math:`(N, C_{in})`。

    输出：
        Tensor，输出张量的形状是 :math:`(N, C_{out}, H_{out}, W_{out})`。
