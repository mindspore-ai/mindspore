mindspore.nn.Conv1dTranspose
=============================

.. py:class:: mindspore.nn.Conv1dTranspose(in_channels, out_channels, kernel_size, stride=1, pad_mode='same', padding=0, dilation=1, group=1, has_bias=False, weight_init='normal', bias_init='zeros')

    一维转置卷积层。
    计算一维转置卷积，也称为反卷积（实际不是真正的反卷积）。
    该算子可以看成Conv1d相对于其输入的梯度。
    `x` 的shape通常是 :math:`(N, C, W)` ，其中 :math:`N` 是batch size， :math:`C` 是通道数， :math:`W` 是特征长度。
    对输入填充 :math:`dilation * (kernel\_size - 1) - padding` 个零。
    因此，当Conv1d和ConvTranspose1d使用相同的参数初始化时，它们的输入和输出shape是互逆的。
    但是，当stride>1时，Conv1d将多个输入的shape映射到同一个输出shape。
    
    输出宽度定义如下：

    .. math::
        W_{out} = \begin{cases}
        (W_{in} - 1) \times \text{stride} - 2 \times \text{padding} + \text{dilation} \times
        (\text{ks_w} - 1) + 1, & \text{if pad_mode='pad'}\\
        (W_{in} + \text{stride} - 1)/\text{stride}, & \text{if pad_mode='same'}\\
        (W_{in} - 1) \times \text{stride} + \text{dilation} \times
        (\text{ks_w} - 1) + 1, & \text{if pad_mode='valid'}
        \end{cases}

    其中 :math:`\text{ks_w}` 是卷积核的宽度。

    **参数：**

    - **in_channels** (int) - 输入的通道数。
    - **out_channels** (int) - 输出的通道数。
    - **kernel_size** (int) - 指定一维卷积窗口的宽度。
    - **stride** (int) - 步长大小，表示移到宽度。默认值：1。
    - **pad_mode** (str) - 选择填充模式。可选值为"pad"，"same"，"valid"。默认值："same"。

      - same：采用补全方式。输出的宽度与输入 `x` 一致。填充总数将在水平上进行计算。并尽可能均匀分布到左侧和右侧。否则，最后一次将从底部和右侧进行额外的填充。若设置该模式， `padding` 必须为0。
      - valid：采用的丢弃方式。在不填充的前提下返回可能大的宽度的输出。多余的像素会被丢弃。如果设置此模式，则 `padding` 必须为0。
      - pad：输入 `x` 两侧的隐式填充。 `padding` 的数量将填充到输入Tensor边框上。 `padding` 必须大于或等于0。

    - **padding** (int) - 输入`x`两侧的隐式填充。默认值：0。
    - **dilation** (int) - 指定用于扩张卷积的扩张速率。如果设置为 :math:`k > 1` ，则每个采样位置都跳过 :math:`k - 1` 个像素。其值必须大于或等于1，并以输入 `x` 的宽度为界。默认值：1。
    - **group** (int) - 将过滤器拆分为组， `in_ channels` 和 `out_channels` 必须可被组数整除。当组数>1时，不支持Davinci设备。默认值：1。
    - **has_bias** (bool) - 指定图层是否使用偏置矢量。默认值：False。
    - **weight_init** (`Union[Tensor, str, Initializer, numbers.Number]`) – 卷积核的初始化方法。它可以是Tensor，str，初始化实例或numbers.Number。当使用str时，可选“TruncatedNormal”，“Normal”，“Uniform”，“HeUniform”和“XavierUniform”分布以及常量“One”和“Zero”分布的值，可接受别名“ xavier_uniform”，“ he_uniform”，“ ones”和“ zeros”。上述字符串大小写均可。更多细节请参考Initializer的值。默认值：“normal”。
    - **bias_init** (`Union[Tensor, str, Initializer, numbers.Number]`) – 偏置向量的初始化方法。可以使用的初始化方法和字符串与“weight_init”相同。更多细节请参考Initializer的值。默认值：“zeros”。

    **输入：**

    - **x** (Tensor) - shape为 :math:`(N, C_{in}, W_{in})` 的Tensor。

    **输出：**

    Tensor，shape为 :math:`(N, C_{out}, W_{out})` 。

    **异常：**

    - **TypeError** - `in_channels` 、 `out_channels` 、 `kernel_size` 、 `stride` 、 `padding`或 `dilation` 不是int。
    - **ValueError** - `in_channels` 、 `out_channels` 、 `kernel_size` 、 `stride` 或 `dilation` 小于1。
    - **ValueError** - `padding` 小于0。
    - **ValueError** - `pad_mode` 不是'same'，'valid'，或'pad'。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> net = nn.Conv1dTranspose(3, 64, 4, has_bias=False, weight_init='normal', pad_mode='pad')
    >>> x = Tensor(np.ones([1, 3, 50]), mindspore.float32)
    >>> output = net(x).shape
    >>> print(output)
    (1, 64, 53)
    