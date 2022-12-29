mindspore.nn.Conv1dTranspose
=============================

.. py:class:: mindspore.nn.Conv1dTranspose(in_channels, out_channels, kernel_size, stride=1, pad_mode='same', padding=0, dilation=1, group=1, has_bias=False, weight_init='normal', bias_init='zeros')

    计算一维转置卷积，可以视为Conv1d对输入求梯度，也称为反卷积（实际不是真正的反卷积）。

    输入的shape通常是 :math:`(N, C_{in}, L_{in})` ，其中 :math:`N` 是batch size， :math:`C_{in}` 是空间维度， :math:`L_{in}` 是序列的长度。
    当Conv1d和ConvTranspose1d使用相同的参数初始化时，且 `pad_mode` 设置为"pad"，它们会在输入的两端填充 :math:`dilation * (kernel\_size - 1) - padding` 个零，这种情况下它们的输入和输出shape是互逆的。
    然而，当 `stride` 大于1时，Conv1d会将多个输入的shape映射到同一个输出shape。反卷积网络可以参考 `Deconvolutional Networks <https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf>`_ 。

    参数：
        - **in_channels** (int) - Conv1dTranspose层输入Tensor的空间维度。
        - **out_channels** (int) - Conv1dTranspose层输出Tensor的空间维度。
        - **kernel_size** (int) - 指定一维卷积核的宽度。
        - **stride** (int) - 一维卷积核的移动步长，默认值：1。
        - **pad_mode** (str) - 指定填充模式。可选值为"same"、"valid"、"pad"。默认值："same"。

          - same：输出的宽度与输入整除 `stride` 后的值相同。若设置该模式， `padding` 的值必须为0。
          - valid：在不填充的前提下返回有效计算所得的输出。不满足计算的多余像素会被丢弃。如果设置此模式，则 `padding` 的值必须为0。
          - pad：对输入进行填充。在输入对两侧填充 `padding` 大小的0。如果设置此模式， `padding` 必须大于或等于0。

        - **padding** (int) - 输入两侧填充的数量。默认值：0。
        - **dilation** (int) - 一维卷积核膨胀尺寸。若 :math:`k > 1` ，则kernel间隔 `k` 个元素进行采样。 `k` 取值范围为[1, L]。默认值：1。
        - **group** (int) - 将过滤器拆分为组， `in_channels` 和 `out_channels` 必须可被 `group` 整除。当 `group` 大于1时，暂不支持Ascend平台。默认值：1。
        - **has_bias** (bool) - Conv1dTranspose层是否添加偏置参数。默认值：False。
        - **weight_init** (Union[Tensor, str, Initializer, numbers.Number]) - 权重参数的初始化方法。它可以是Tensor，str，Initializer或numbers.Number。当使用str时，可选"TruncatedNormal"，"Normal"，"Uniform"，"HeUniform"和"XavierUniform"分布以及常量"One"和"Zero"分布的值，可接受别名"xavier_uniform"，"he_uniform"，"ones"和"zeros"。上述字符串大小写均可。更多细节请参考Initializer的值。默认值："normal"。
        - **bias_init** (Union[Tensor, str, Initializer, numbers.Number]) - 偏置参数的初始化方法。可以使用的初始化方法与"weight_init"相同。更多细节请参考Initializer的值。默认值："zeros"。

    输入：
        - **x** (Tensor) - shape为 :math:`(N, C_{in}, L_{in})` 的Tensor。

    输出：
        Tensor，shape为 :math:`(N, C_{out}, L_{out})` 。

        当 `pad_mode` 设置为"same"时：

        .. math::
            L_{out} = \frac{ L_{in} + \text{stride} - 1 }{ \text{stride} }

        当 `pad_mode` 设置为"valid"时：

        .. math::
            L_{out} = (L_{in} - 1) \times \text{stride} + \text{dilation} \times (\text{kernel_size} - 1) + 1

        当 `pad_mode` 设置为"pad"时：

        .. math::
            L_{out} = (L_{in} - 1) \times \text{stride} - 2 \times \text{padding} + \text{dilation} \times (\text{kernel_size} - 1) + 1

    异常：
        - **TypeError** - `in_channels` 、 `out_channels` 、 `kernel_size` 、 `stride` 、 `padding` 或 `dilation` 不是int。
        - **ValueError** - `in_channels` 、 `out_channels` 、 `kernel_size` 、 `stride` 或 `dilation` 小于1。
        - **ValueError** - `padding` 小于0。
        - **ValueError** - `pad_mode` 不是"same"，"valid"或"pad"。
