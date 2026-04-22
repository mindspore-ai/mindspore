mindspore.nn.Conv3dTranspose
=============================

.. py:class:: mindspore.nn.Conv3dTranspose(in_channels, out_channels, kernel_size, stride=1, pad_mode='same', padding=0, dilation=1, group=1, output_padding=0, has_bias=False, weight_init=None, bias_init=None, data_format='NCDHW', dtype=mstype.float32)

    计算三维转置卷积，可以视为Conv3d对输入求梯度，也称为反卷积（实际不是真正的反卷积）。

    输入的shape通常为 :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` ，其中 :math:`N` 为batch size， :math:`C_{in}` 是空间维度，:math:`D_{in}, H_{in}, W_{in}` 分别为特征层的深度、高度和宽度。
    当Conv3d和ConvTranspose3d使用相同的参数初始化时，且 `pad_mode` 设置为 ``"pad"``，它们会在输入的深度、高度和宽度方向上填充 :math:`dilation * (kernel\_size - 1) - padding` 个0，这种情况下它们的输入shape和输出shape是互逆的。
    然而，当 `stride` 大于1时，Conv3d会将多个输入的shape映射到同一个输出shape。反卷积网络的详细介绍可以参考论文： `Deconvolutional Networks <https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf>`_ 。

    .. warning::
        从2.9.0（不含）之后版本开始，构造签名将调整为 ``Conv3dTranspose(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', dtype=None)`` 。其中参数名 `group` 将改为 `groups`，`has_bias` 将改为 `bias` 且默认值由 ``False`` 改为 ``True`` ；`pad_mode`、`weight_init`、`bias_init`、`data_format` 将被移除；新增参数 `padding_mode`。

    参数：
        - **in_channels** (int) - Conv3dTranspose层输入Tensor的空间维度。
        - **out_channels** (int) - Conv3dTranspose层输出Tensor的空间维度。
        - **kernel_size** (Union[int, tuple[int]]) - 指定三维卷积核的深度、高度和宽度。数据类型为int，或包含三个整数的tuple。若为一个整数，则表示卷积核的深度、高度和宽度均为该整数值；若为包含三个整数的tuple，则分别表示卷积核的深度、高度和宽度。
        - **stride** (Union[int, tuple[int]]，可选) - 三维卷积核的移动步长。数据类型为int，或包含三个整型的tuple。若为一个整数，则表示在深度、高度和宽度方向的移动步长均为该整数值；若为包含三个整数的tuple，则分别表示在深度、高度和宽度方向的移动步长。默认值： ``1`` 。
        - **pad_mode** (str，可选) - 指定填充模式，填充值为0。可选值为 ``"same"`` ， ``"valid"`` 或 ``"pad"`` 。默认值： ``"same"`` 。

          - ``"same"``：在输入的深度、高度和宽度维度进行填充，使得当 `stride` 为 ``1`` 时，输入和输出的shape一致。待填充的量由算子内部计算，若为偶数，则均匀地填充在四周；若为奇数，则多余的填充量将补充在前方/底部/右侧。如果设置了此模式， `padding` 必须为0。
          - ``"valid"``：不对输入进行填充，返回输出可能的最大深度、高度和宽度，如果不能构成一个完整stride，那么额外的像素将被丢弃。如果设置了此模式， `padding` 必须为0。
          - ``"pad"``：对输入填充指定的量。在这种模式下，在输入的深度、高度和宽度方向上填充的量由 `padding` 参数指定。如果设置此模式， `padding` 必须大于或等于0。

        - **padding** (Union(int, tuple[int])，可选) - 输入的深度、高度和宽度方向上填充的数量。数据类型为int，或包含六个整数的tuple。若 `padding` 是一个整数，则前部、后部、顶部，底部，左边和右边的填充都等于 `padding` ；若 `padding` 是包含六个整数的tuple，则前部、尾部、顶部、底部、左边和右边的填充分别等于填充padding[0]、padding[1]、padding[2]、padding[3]、padding[4]和padding[5]。值要大于等于0，默认值： ``0`` 。
        - **dilation** (Union[int, tuple[int]]，可选) - 三维卷积核的膨胀尺寸。数据类型为int，或包含三个整数的tuple。若为一个整数，则表示在深度、高度和宽度方向的膨胀尺寸均为该整数值；若为包含三个整数的tuple，则分别表示在深度、高度和宽度方向的膨胀尺寸。
          假设 :math:`dilation=(d0, d1, d2)`, 则卷积核在深度方向间隔 :math:`d0-1` 个元素进行采样，在高度方向间隔 :math:`d1-1` 个元素进行采样，在宽度方向间隔 :math:`d2-1` 个元素进行采样。深度、高度和宽度上取值范围分别为[1, D]、[1, H]和[1, W]。默认值： ``1`` 。
        - **group** (int，可选) - 将过滤器拆分为组， `in_channels` 和 `out_channels` 必须可被 `group` 整除。当 `group` 大于1时，暂不支持Ascend平台。默认值： ``1`` 。
        - **output_padding** (Union(int, tuple[int])，可选) - 输出的深度、高度和宽度方向上填充的数量。数据类型为int，或包含3个整数的tuple。如果 `output_padding` 是一个整数，则深度、高度和宽度方向的填充都等于 `output_padding` ；如果 `output_padding` 是包含三个整数的tuple，则深度、高度和宽度方向的填充分别等于填充output_padding[0]、output_padding[1]和output_padding[2]。值要大于等于0，默认值： ``0`` 。
        - **has_bias** (bool，可选) - Conv3dTranspose层是否添加偏置参数。默认值： ``False`` 。
        - **weight_init** (Union[Tensor, str, Initializer, numbers.Number]，可选) - 权重参数的初始化方法。它可以是Tensor、str、Initializer或numbers.Number。当使用str时，可选 ``"TruncatedNormal"`` ， ``"Normal"`` ， ``"Uniform"`` ， ``"HeUniform"`` 和 ``"XavierUniform"`` 分布以及常量 ``"One"`` 和 ``"Zero"`` 分布的值，可接受别名 ``"xavier_uniform"`` ， ``"he_uniform"`` ， ``"ones"`` 和 ``"zeros"`` 。上述字符串大小写均可。更多细节请参考Initializer的值。默认值： ``None`` ，即权重使用HeUniform初始化。
        - **bias_init** (Union[Tensor, str, Initializer, numbers.Number]，可选) - 偏置参数的初始化方法。可以使用的初始化方法与 `weight_init` 相同。更多细节请参考Initializer的值。默认值： ``None`` ，即偏差使用Uniform初始化。
        - **data_format** (str，可选) - 数据格式的可选值。目前仅支持 ``'NCDHW'`` 。 默认值： ``'NCDHW'`` 。
        - **dtype** (:class:`mindspore.dtype`，可选) - Parameters的dtype，需要跟输入的dtype保持一致。默认值： ``mstype.float32`` 。

    输入：
        - **x** (Tensor) - shape为 :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` 的Tensor。目前，CPU/GPU只支持输入数据类型为float16和float32，Ascend只支持输入数据类型为float16。

    输出：
        Tensor，shape为 :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` 。

        pad_mode为 ``"same"`` 时：

        .. math::
            \begin{array}{ll} \\
                D_{out} = \left \lfloor{\frac{D_{in}}{\text{stride[0]}} + 1} \right \rfloor \\
                H_{out} = \left \lfloor{\frac{H_{in}}{\text{stride[1]}} + 1} \right \rfloor \\
                W_{out} = \left \lfloor{\frac{W_{in}}{\text{stride[2]}} + 1} \right \rfloor \\
            \end{array}

        pad_mode为 ``"valid"`` 时：

        .. math::
            \begin{array}{ll} \\
                D_{out} = \left \lfloor{\frac{D_{in} - \text{dilation[0]} \times (\text{kernel_size[0]} - 1) }
                {\text{stride[0]}} + 1} \right \rfloor \\
                H_{out} = \left \lfloor{\frac{H_{in} - \text{dilation[1]} \times (\text{kernel_size[1]} - 1) }
                {\text{stride[1]}} + 1} \right \rfloor \\
                W_{out} = \left \lfloor{\frac{W_{in} - \text{dilation[2]} \times (\text{kernel_size[2]} - 1) }
                {\text{stride[2]}} + 1} \right \rfloor \\
            \end{array}

        pad_mode为 ``"pad"`` 时：

        .. math::
            \begin{array}{ll} \\
                D_{out} = \left \lfloor{\frac{D_{in} + padding[0] + padding[1] - (\text{dilation[0]} - 1) \times
                \text{kernel_size[0]} - 1 }{\text{stride[0]}} + 1} \right \rfloor \\
                H_{out} = \left \lfloor{\frac{H_{in} + padding[2] + padding[3] - (\text{dilation[1]} - 1) \times
                \text{kernel_size[1]} - 1 }{\text{stride[1]}} + 1} \right \rfloor \\
                W_{out} = \left \lfloor{\frac{W_{in} + padding[4] + padding[5] - (\text{dilation[2]} - 1) \times
                \text{kernel_size[2]} - 1 }{\text{stride[2]}} + 1} \right \rfloor \\
            \end{array}

    异常：
        - **TypeError** - `in_channels` 、 `out_channels` 或 `group` 不是int。
        - **TypeError** - `kernel_size` 、 `stride` 、 `padding` 、 `dilation` 或 `output_padding` 既不是int也不是tuple。
        - **TypeError** - 输入数据类型不是要求的类型，即CPU/GPU不是float16或float32，Ascend不是float16。
        - **ValueError** - `in_channels` 、 `out_channels` 、 `kernel_size` 、 `stride` 或 `dilation` 小于1。
        - **ValueError** - `padding` 小于0。
        - **ValueError** - `pad_mode` 不是 ``"same"`` ， ``"valid"`` 或 ``"pad"`` 。
        - **ValueError** - `padding` 是长度不等于6的tuple。
        - **ValueError** - `pad_mode` 不等于 ``"pad"`` 且 `padding` 不等于 ``(0, 0, 0, 0, 0, 0)``。
        - **ValueError** - `data_format` 不是 ``"NCDHW"``。
