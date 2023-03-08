mindspore.nn.Conv2dTranspose
============================

.. py:class:: mindspore.nn.Conv2dTranspose(in_channels, out_channels, kernel_size, stride=1, pad_mode="same", padding=0, output_padding=0, dilation=1, group=1, has_bias=False, weight_init="normal", bias_init="zeros")

    计算二维转置卷积，可以视为Conv2d对输入求梯度，也称为反卷积（实际不是真正的反卷积）。

    输入的shape通常为 :math:`(N, C_{in}, H_{in}, W_{in})` ，其中 :math:`N` 是batch size，:math:`C_{in}` 是空间维度， :math:`H_{in}, W_{in}` 分别为特征层的高度和宽度。
    当Conv2d和ConvTranspose2d使用相同的参数初始化时，且 `pad_mode` 设置为"pad"，它们会在输入的高度和宽度方向上填充 :math:`dilation * (kernel\_size - 1) - padding` 个零，这种情况下它们的输入和输出shape是互逆的。
    然而，当 `stride` 大于1时，Conv2d会将多个输入的shape映射到同一个输出shape。反卷积网络可以参考 `Deconvolutional Networks <https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf>`_ 。

    参数：
        - **in_channels** (int) - Conv2dTranspose层输入Tensor的空间维度。
        - **out_channels** (int) - Conv2dTranspose层输出Tensor的空间维度。
        - **kernel_size** (Union[int, tuple[int]]) - 指定二维卷积核的高度和宽度。数据类型为整型或两个整型的tuple。一个整数表示卷积核的高度和宽度均为该值。两个整数的tuple分别表示卷积核的高度和宽度。
        - **stride** (Union[int, tuple[int]]) - 二维卷积核的移动步长。数据类型为整型或两个整型的tuple。一个整数表示在高度和宽度方向的移动步长均为该值。两个整数的tuple分别表示在高度和宽度方向的移动步长。默认值：1。
        - **pad_mode** (str) - 指定填充模式。可选值为"same"、"valid"、"pad"。默认值："same"。

          - **same**：输出的高度和宽度分别与输入整除 `stride` 后的值相同。若设置该模式，`padding` 的值必须为0。
          - **valid**：在不填充的前提下返回有效计算所得的输出。不满足计算的多余像素会被丢弃。如果设置此模式，则 `padding` 的值必须为0。
          - **pad**：对输入进行填充。在输入的高度和宽度方向上填充 `padding` 大小的0。如果设置此模式， `padding` 必须大于或等于0。

        - **padding** (Union[int, tuple[int]]) - 输入的高度和宽度方向上填充的数量。数据类型为整型或包含四个整数的tuple。如果 `padding` 是一个整数，那么上、下、左、右的填充都等于 `padding` 。如果 `padding` 是一个有四个整数的tuple，那么上、下、左、右的填充分别等于 `padding[0]` 、 `padding[1]` 、 `padding[2]` 和 `padding[3]` 。值应该要大于等于0，默认值：0。
        - **output_padding** (Union[int, tuple[int]]) - 输出的高度和宽度方向上填充的数量。数据类型为整型或包含两个整数的tuple。如果 `output_padding` 是一个整数，那么下、右的填充都等于 `output_padding` 。如果 `output_padding` 是一个有两个整数的tuple，那么下、右的填充分别等于 `output_padding[0]` 、 `output_padding[1]` 。如果 `output_padding` 不为0， `pad_mode` 必须为 `pad` 。 `output_padding` 取值范围为 `[0, max(stride, dilation))` ，默认值：0。
        - **dilation** (Union[int, tuple[int]]) - 二维卷积核膨胀尺寸。数据类型为整型或具有两个整型的tuple。若 :math:`k > 1` ，则kernel间隔 `k` 个元素进行采样。高度和宽度方向上的 `k` ，其取值范围分别为[1, H]和[1, W]。默认值：1。
        - **group** (int) - 将过滤器拆分为组， `in_channels` 和 `out_channels` 必须可被 `group` 整除。如果组数等于 `in_channels` 和 `out_channels` ，这个二维卷积层也被称为二维深度卷积层。默认值：1.
        - **has_bias** (bool) - Conv2dTranspose层是否添加偏置参数。默认值：False。
        - **weight_init** (Union[Tensor, str, Initializer, numbers.Number]) - 权重参数的初始化方法。它可以是Tensor，str，Initializer或numbers.Number。当使用str时，可选"TruncatedNormal"，"Normal"，"Uniform"，"HeUniform"和"XavierUniform"分布以及常量"One"和"Zero"分布的值，可接受别名"xavier_uniform"，"he_uniform"，"ones"和"zeros"。上述字符串大小写均可。更多细节请参考Initializer的值。默认值："normal"。
        - **bias_init** (Union[Tensor, str, Initializer, numbers.Number]) - 偏置参数的初始化方法。可以使用的初始化方法与"weight_init"相同。更多细节请参考Initializer的值。默认值："zeros"。

    输入：
        - **x** (Tensor) - Shape 为 :math:`(N, C_{in}, H_{in}, W_{in})` 的Tensor。

    输出：
        Tensor，shape为 :math:`(N, C_{out}, H_{out}, W_{out})` 。

        pad_mode为"same"时：

        .. math::
            \begin{array}{ll} \\
                H_{out} = \text H_{in}\times \text {stride[0]} \\
                W_{out} = \text W_{in}\times \text {stride[1]} \\
            \end{array}

        pad_mode为"valid"时：

        .. math::
            \begin{array}{ll} \\
                H_{out} = \text H_{in}\times \text {stride[0]} + \max\{(\text{dilation[0]} - 1) \times
                (\text{kernel_size[0]} - 1) - \text {stride[0]}, 0 \} \\
                W_{out} = \text W_{in}\times \text {stride[1]} + \max\{(\text{dilation[1]} - 1) \times
                (\text{kernel_size[1]} - 1) - \text {stride[1]}, 0 \} \\
            \end{array}

        pad_mode为"pad"时：

        .. math::
            \begin{array}{ll} \\
                H_{out} = \text H_{in}\times \text {stride[0]} - (padding[0] + padding[1]) + \text{kernel_size[0]} + (\text{dilation[0]} - 1) \times
                (\text{kernel_size[0]} - 1) - \text {stride[0]} + \text {output_padding[0]} \\
                W_{out} = \text W_{in}\times \text {stride[1]} - (padding[2] + padding[3]) + \text{kernel_size[1]} + (\text{dilation[1]} - 1) \times
                (\text{kernel_size[1]} - 1) - \text {stride[1]} + \text {output_padding[1]} \\
            \end{array}

    异常：
        - **TypeError** - 如果 `in_channels` ，`out_channels` 或者 `group` 不是整数。
        - **TypeError** - 如果 `kernel_size` ，`stride` ，`padding` 或者 `dilation` 既不是整数也不是tuple。
        - **ValueError** - 如果 `in_channels` ，`out_channels` ， `kernel_size` ， `stride` 或者 `dilation` 小于1。
        - **ValueError** - 如果 `padding` 小于0。
        - **ValueError** - 如果 `pad_mode` 不是"same"，"valid"或"pad"。
        - **ValueError** - 如果 `padding` 是一个长度不等于4的tuple。
        - **ValueError** - 如果 `pad_mode` 不等于"pad"且 `padding` 不等于(0,0,0,0)。
