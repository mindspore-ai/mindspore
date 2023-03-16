mindspore.nn.Conv2d
====================

.. py:class:: mindspore.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, pad_mode="same", padding=0, dilation=1, group=1, has_bias=False, weight_init="normal", bias_init="zeros", data_format="NCHW")

    对输入Tensor计算二维卷积。该Tensor的常见shape为 :math:`(N, C_{in}, H_{in}, W_{in})`，其中 :math:`N` 为batch size，:math:`C_{in}` 为空间维度，:math:`H_{in}, W_{in}` 分别为特征层的高度和宽度。对于每个batch中的Tensor，其shape为 :math:`(C_{in}, H_{in}, W_{in})` ，公式定义如下：

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{ccor}({\text{weight}(C_{\text{out}_j}, k), \text{X}(N_i, k)})

    其中，:math:`ccor` 为 `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_ ， :math:`C_{in}` 为输入空间维度，  :math:`out_{j}` 对应输出的第 :math:`j` 个空间维度，:math:`j` 的范围在 :math:`[0, C_{out}-1]` 内。
    :math:`\text{weight}(C_{\text{out}_j}, k)` 是shape为 :math:`(\text{kernel_size[0]}, \text{kernel_size[1]})` 的卷积核切片，其中  :math:`\text{kernel_size[0]}` 和 :math:`\text{kernel_size[1]}` 分别是卷积核的高度和宽度。 :math:`\text{bias}` 为偏置参数， :math:`\text{X}` 为输入Tensor。
    此时，输入Tensor对应的 `data_format` 为"NCHW"，完整卷积核的shape为 :math:`(C_{out}, C_{in} / \text{group}, \text{kernel_size[0]}, \text{kernel_size[1]})` ，其中 `group` 是在空间维度上分割输入 `x` 的组数。如果输入Tensor对应的 `data_format` 为"NHWC"，完整卷积核的shape则为 :math:`(C_{out}, \text{kernel_size[0]}, \text{kernel_size[1]}), C_{in} / \text{group}`。
    详细介绍请参考论文 `Gradient Based Learning Applied to Document Recognition <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_ 。

    .. note::
        在Ascend平台上，目前只支持深度卷积场景下的分组卷积运算。也就是说，当 `group>1` 的场景下，必须要满足 `in\_channels` = `out\_channels` = `group` 的约束条件。

    参数：
        - **in_channels** (int) - Conv2d层输入Tensor的空间维度。
        - **out_channels** (int) - Conv2d层输出Tensor的空间维度。
        - **kernel_size** (Union[int, tuple[int]]) - 指定二维卷积核的高度和宽度。数据类型为整型或两个整型的tuple。一个整数表示卷积核的高度和宽度均为该值。两个整数的tuple分别表示卷积核的高度和宽度。
        - **stride** (Union[int, tuple[int]]) - 二维卷积核的移动步长。数据类型为整型或两个整型的tuple。一个整数表示在高度和宽度方向的移动步长均为该值。两个整数的tuple分别表示在高度和宽度方向的移动步长。默认值：1。
        - **pad_mode** (str) - 指定填充模式。可选值为"same"、"valid"、"pad"。默认值："same"。

          - **same**：输出的高度和宽度分别与输入整除 `stride` 后的值相同。若设置该模式，`padding` 的值必须为0。
          - **valid**：在不填充的前提下返回有效计算所得的输出。不满足计算的多余像素会被丢弃。如果设置此模式，则 `padding` 的值必须为0。
          - **pad**：对输入进行填充。在输入的高度和宽度方向上填充 `padding` 大小的0。如果设置此模式， `padding` 必须大于或等于0。

        - **padding** (Union[int, tuple[int]]) - 输入的高度和宽度方向上填充的数量。数据类型为int或包含4个整数的tuple。如果 `padding` 是一个整数，那么上、下、左、右的填充都等于 `padding` 。如果 `padding` 是一个有4个整数的tuple，那么上、下、左、右的填充分别等于 `padding[0]` 、 `padding[1]` 、 `padding[2]` 和 `padding[3]` 。值应该要大于等于0，默认值：0。
        - **dilation** (Union[int, tuple[int]]) - 二维卷积核膨胀尺寸。数据类型为整型或具有两个整型的tuple。若 :math:`k > 1` ，则kernel间隔 `k` 个元素进行采样。垂直和水平方向上的 `k` ，其取值范围分别为[1, H]和[1, W]。默认值：1。
        - **group** (int) - 将过滤器拆分为组， `in_channels` 和 `out_channels` 必须可被 `group` 整除。如果组数等于 `in_channels` 和 `out_channels` ，这个二维卷积层也被称为二维深度卷积层。默认值：1.
        - **has_bias** (bool) - Conv2d层是否添加偏置参数。默认值：False。
        - **weight_init** (Union[Tensor, str, Initializer, numbers.Number]) - 权重参数的初始化方法。它可以是Tensor，str，Initializer或numbers.Number。当使用str时，可选"TruncatedNormal"，"Normal"，"Uniform"，"HeUniform"和"XavierUniform"分布以及常量"One"和"Zero"分布的值，可接受别名"xavier_uniform"，"he_uniform"，"ones"和"zeros"。上述字符串大小写均可。更多细节请参考Initializer的值。默认值："normal"。
        - **bias_init** (Union[Tensor, str, Initializer, numbers.Number]) - 偏置参数的初始化方法。可以使用的初始化方法与"weight_init"相同。更多细节请参考Initializer的值。默认值："zeros"。
        - **data_format** (str) - 数据格式的可选值有"NHWC"，"NCHW"。默认值："NCHW"。

    输入：
        - **x** (Tensor) - Shape为 :math:`(N, C_{in}, H_{in}, W_{in})` 或者 :math:`(N, H_{in}, W_{in}, C_{in})` 的Tensor。

    输出：
        Tensor，shape为 :math:`(N, C_{out}, H_{out}, W_{out})` 或者 :math:`(N, H_{out}, W_{out}, C_{out})` 。

        pad_mode为"same"时：

        .. math::
            \begin{array}{ll} \\
                H_{out} ＝ \left \lceil{\frac{H_{in}}{\text{stride[0]}}} \right \rceil \\
                W_{out} ＝ \left \lceil{\frac{W_{in}}{\text{stride[1]}}} \right \rceil \\
            \end{array}

        pad_mode为"valid"时：

        .. math::
            \begin{array}{ll} \\
                H_{out} ＝ \left \lceil{\frac{H_{in} - \text{dilation[0]} \times (\text{kernel_size[0]} - 1) }
                {\text{stride[0]}}} \right \rceil \\
                W_{out} ＝ \left \lceil{\frac{W_{in} - \text{dilation[1]} \times (\text{kernel_size[1]} - 1) }
                {\text{stride[1]}}} \right \rceil \\
            \end{array}

        pad_mode为"pad"时：

        .. math::
            \begin{array}{ll} \\
                H_{out} ＝ \left \lfloor{\frac{H_{in} + padding[0] + padding[1] - (\text{kernel_size[0]} - 1) \times
                \text{dilation[0]} - 1 }{\text{stride[0]}} + 1} \right \rfloor \\
                W_{out} ＝ \left \lfloor{\frac{W_{in} + padding[2] + padding[3] - (\text{kernel_size[1]} - 1) \times
                \text{dilation[1]} - 1 }{\text{stride[1]}} + 1} \right \rfloor \\
            \end{array}

    异常：
        - **TypeError** - 如果 `in_channels` ， `out_channels` 或者 `group` 不是整数。
        - **TypeError** - 如果 `kernel_size` ， `stride`，`padding` 或者 `dilation` 既不是整数也不是tuple。
        - **ValueError** - 如果 `in_channels` ， `out_channels`，`kernel_size` ， `stride` 或者 `dilation` 小于1。
        - **ValueError** - 如果 `padding` 小于0。
        - **ValueError** - 如果 `pad_mode` 不是"same"，"valid"或"pad"。
        - **ValueError** - 如果 `padding` 是一个长度不等于4的tuple。
        - **ValueError** - 如果 `pad_mode` 不等于"pad"且 `padding` 不等于(0,0,0,0)。
        - **ValueError** - 如果 `data_format` 既不是"NCHW"也不是"NHWC"。
