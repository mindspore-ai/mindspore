mindspore.nn.Conv3dTranspose
=============================

.. py:class:: mindspore.nn.Conv3dTranspose(in_channels, out_channels, kernel_size, stride=1, pad_mode='same', padding=0, dilation=1, group=1, output_padding=0, has_bias=False, weight_init='normal', bias_init='zeros', data_format='NCDHW')

    三维转置卷积层。

    计算三维转置卷积，可以视为Conv3d对输入求梯度，也称为反卷积（实际不是真正的反卷积）。

    输入的shape通常为 :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` ，其中 :math:`N` 为batch size， :math:`C_{in}` 是空间维度。:math:`D_{in}, H_{in}, W_{in}` 分别为特征层的深度、高度和宽度。
    当Conv3d和ConvTranspose3d使用相同的参数初始化时，且 `pad_mode` 设置为"pad"，它们会在输入的深度、高度和宽度方向上填充 :math:`dilation * (kernel\_size - 1) - padding` 个零，这种情况下它们的输入和输出shape是互逆的。
    然而，当 `stride` 大于1时，Conv3d会将多个输入的shape映射到同一个输出shape。反卷积网络可以参考 `Deconvolutional Networks <https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf>`_ 。

    参数：
        - **in_channels** (int) - Conv3dTranspose层输入Tensor的空间维度。
        - **out_channels** (int) - Conv3dTranspose层输出Tensor的空间维度。
        - **kernel_size** (Union[int, tuple[int]]) - 指定三维卷积核的深度、高度和宽度。数据类型为int或包含三个整数的tuple。一个整数表示卷积核的深度、高度和宽度均为该值该值。包含三个整数的tuple分别表示卷积核的深度、高度和宽度。
        - **stride** (Union[int, tuple[int]]) - 三维卷积核的移动步长。数据类型为整型或三个整型的tuple。一个整数表示在深度、高度和宽度方向的移动步长均为该值。三个整数的tuple分别表示在深度、高度和宽度方向的移动步长。默认值：1。
        - **pad_mode** (str) - 指定填充模式。可选值为"same"、"valid"、"pad"。默认值："same"。

          - same：输出的深度、高度和宽度分别与输入整除 `stride` 后的值相同。若设置该模式，`padding` 的值必须为0。
          - valid：在不填充的前提下返回有效计算所得的输出。不满足计算的多余像素会被丢弃。如果设置此模式，则 `padding` 的值必须为0。
          - pad：对输入进行填充。在输入的深度、高度和宽度方向上填充 `padding` 大小的0。如果设置此模式， `padding` 必须大于或等于0。

        - **padding** (Union(int, tuple[int])) - 输入的深度、高度和宽度方向上填充的数量。数据类型为int或包含6个整数的tuple。如果 `padding` 是一个整数，则前部、后部、顶部，底部，左边和右边的填充都等于 `padding` 。如果 `padding` 是6个整数的tuple，则前部、尾部、顶部、底部、左边和右边的填充分别等于填充padding[0]、padding[1]、padding[2]、padding[3]、padding[4]和padding[5]。值应该要大于等于0，默认值：0。
        - **dilation** (Union[int, tuple[int]]) - 三维卷积核膨胀尺寸。数据类型为int或三个整数的tuple。若 :math:`k > 1` ，则kernel间隔 `k` 个元素进行采样。深度、高度和宽度方向上的 `k` ，其取值范围分别为[1, D]、[1, H]和[1, W]。默认值：1。
        - **group** (int) - 将过滤器拆分为组， `in_channels` 和 `out_channels` 必须可被 `group` 整除。当 `group` 大于1时，暂不支持Ascend平台。默认值：1。当前仅支持1。
        - **output_padding** (Union(int, tuple[int])) - 输出的深度、高度和宽度方向上填充的数量。数据类型为int或包含6个整数的tuple。如果 `output_padding` 是一个整数，则前部、后部、顶部，底部，左边和右边的填充都等于 `output_padding` 。如果 `output_padding` 是6个整数的tuple，则前部、尾部、顶部、底部、左边和右边的填充分别等于填充output_padding[0]、output_padding[1]、output_padding[2]、output_padding[3]、output_padding[4]output_padding[5]。值应该要大于等于0，默认值：0。
        - **has_bias** (bool) - Conv3dTranspose层是否添加偏置参数。默认值：False。
        - **weight_init** (Union[Tensor, str, Initializer, numbers.Number]) - 权重参数的初始化方法。它可以是Tensor，str，Initializer或numbers.Number。当使用str时，可选"TruncatedNormal"，"Normal"，"Uniform"，"HeUniform"和"XavierUniform"分布以及常量"One"和"Zero"分布的值，可接受别名"xavier_uniform"，"he_uniform"，"ones"和"zeros"。上述字符串大小写均可。更多细节请参考Initializer的值。默认值："normal"。
        - **bias_init** (Union[Tensor, str, Initializer, numbers.Number]) - 偏置参数的初始化方法。可以使用的初始化方法与"weight_init"相同。更多细节请参考Initializer的值。默认值："zeros"。
        - **data_format** (str) - 数据格式的可选值。目前仅支持"NCDHW"。

    输入：
        - **x** (Tensor) - shape为 :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` 的Tensor。目前输入数据类型只支持float16和float32。

    输出：
        Tensor，shape为 :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` 。

        pad_mode为"same"时：

        .. math::
            \begin{array}{ll} \\
                D_{out} ＝ \left \lfloor{\frac{D_{in}}{\text{stride[0]}} + 1} \right \rfloor \\
                H_{out} ＝ \left \lfloor{\frac{H_{in}}{\text{stride[1]}} + 1} \right \rfloor \\
                W_{out} ＝ \left \lfloor{\frac{W_{in}}{\text{stride[2]}} + 1} \right \rfloor \\
            \end{array}

        pad_mode为"valid"时：

        .. math::
            \begin{array}{ll} \\
                D_{out} ＝ \left \lfloor{\frac{D_{in} - \text{dilation[0]} \times (\text{kernel_size[0]} - 1) }
                {\text{stride[0]}} + 1} \right \rfloor \\
                H_{out} ＝ \left \lfloor{\frac{H_{in} - \text{dilation[1]} \times (\text{kernel_size[1]} - 1) }
                {\text{stride[1]}} + 1} \right \rfloor \\
                W_{out} ＝ \left \lfloor{\frac{W_{in} - \text{dilation[2]} \times (\text{kernel_size[2]} - 1) }
                {\text{stride[2]}} + 1} \right \rfloor \\
            \end{array}

        pad_mode为"pad"时：

        .. math::
            \begin{array}{ll} \\
                D_{out} ＝ \left \lfloor{\frac{D_{in} + padding[0] + padding[1] - (\text{dilation[0]} - 1) \times
                \text{kernel_size[0]} - 1 }{\text{stride[0]}} + 1} \right \rfloor \\
                H_{out} ＝ \left \lfloor{\frac{H_{in} + padding[2] + padding[3] - (\text{dilation[1]} - 1) \times
                \text{kernel_size[1]} - 1 }{\text{stride[1]}} + 1} \right \rfloor \\
                W_{out} ＝ \left \lfloor{\frac{W_{in} + padding[4] + padding[5] - (\text{dilation[2]} - 1) \times
                \text{kernel_size[2]} - 1 }{\text{stride[2]}} + 1} \right \rfloor \\
            \end{array}

    异常：
        - **TypeError** - `in_channels` 、 `out_channels` 或 `group` 不是int。
        - **TypeError** - `kernel_size` 、 `stride` 、 `padding` 、 `dilation` 或 `output_padding` 既不是int也不是tuple。     
        - **TypeError** - 输入数据类型不是float16或float32。
        - **ValueError** - `in_channels` 、 `out_channels` 、 `kernel_size` 、 `stride` 或 `dilation` 小于1。
        - **ValueError** - `padding` 小于0。
        - **ValueError** - `pad_mode` 不是"same"，"valid"或"pad"。
        - **ValueError** - `padding` 是长度不等于6的tuple。
        - **ValueError** - `pad_mode` 不等于"pad"且 `padding` 不等于(0, 0, 0, 0, 0, 0)。
        - **ValueError** - `data_format` 不是"NCDHW"。
