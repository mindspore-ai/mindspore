mindspore.nn.Conv2d
====================

.. py:class:: mindspore.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, pad_mode="same", padding=0, dilation=1, group=1, has_bias=False, weight_init=None, bias_init=None, data_format="NCHW", dtype=mstype.float32)

    二维卷积层。

    对输入Tensor计算二维卷积，通常输入的shape为 :math:`(N, C_{in}, H_{in}, W_{in})` ，其中 :math:`N` 为batch size，:math:`C` 为通道数， :math:`H` 为特征图的高度，:math:`W` 为特征图的宽度。

    根据以下公式计算输出：

    .. math::

        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{ccor}({\text{weight}(C_{\text{out}_j}, k), \text{X}(N_i, k)})

    其中， :math:`bias` 为输出偏置，:math:`ccor` 为 `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_ 操作， 
    :math:`weight` 为卷积核的值， :math:`X` 为输入的特征图。

    :math:`i` 对应batch数，其范围为 :math:`[0, N-1]` ，其中 :math:`N` 为输入batch。

    :math:`j` 对应输出通道，其范围为 :math:`[0, C_{out}-1]` ，其中 :math:`C_{out}` 为输出通道数，该值也等于卷积核的个数。

    :math:`k` 对应输入通道数，其范围为 :math:`[0, C_{in}-1]` ，其中 :math:`C_{in}` 为输入通道数，该值也等于卷积核的通道数。

    因此，上面的公式中， :math:`{bias}(C_{\text{out}_j})` 为第 :math:`j` 个输出通道的偏置， :math:`{weight}(C_{\text{out}_j}, k)` 表示第 :math:`j` 个
    卷积核在第 :math:`k` 个输入通道的卷积核切片， :math:`{X}(N_i, k)` 为特征图第 :math:`i` 个batch第 :math:`k` 个输入通道的切片。
    卷积核shape为 :math:`(\text{kernel_size[0]},\text{kernel_size[1]})` ，其中 :math:`\text{kernel_size[0]}` 和
    :math:`\text{kernel_size[1]}` 是卷积核的高度和宽度。若考虑到输入输出通道以及group，则完整卷积核的shape为
    :math:`(C_{out}, C_{in} / \text{group}, \text{kernel_size[0]}, \text{kernel_size[1]})` ，
    其中 `group` 是分组卷积时在通道上分割输入 `x` 的组数。

    想更深入了解卷积层，请参考论文 `Gradient Based Learning Applied to Document Recognition <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_ 。

    .. note::
        在Ascend平台上，目前只支持深度卷积场景下的分组卷积运算。也就是说，当 `group>1` 的场景下，必须要满足 `in\_channels` = `out\_channels` = `group` 的约束条件。

    参数：
        - **in_channels** (int) - Conv2d层输入Tensor的空间维度。
        - **out_channels** (int) - Conv2d层输出Tensor的空间维度。
        - **kernel_size** (Union[int, tuple[int]]) - 指定二维卷积核的高度和宽度。数据类型为整型或两个整型的tuple。一个整数表示卷积核的高度和宽度均为该值。两个整数的tuple分别表示卷积核的高度和宽度。
        - **stride** (Union[int, tuple[int]]，可选) - 二维卷积核的移动步长。数据类型为整型或者长度为二或四的整型tuple。一个整数表示在高度和宽度方向的移动步长均为该值。两个整数的tuple分别表示在高度和宽度方向的移动步长。默认值： ``1`` 。
        - **pad_mode** (str，可选) - 指定填充模式，填充值为0。可选值为 ``"same"`` ， ``"valid"`` 或 ``"pad"`` 。默认值： ``"same"`` 。

          - ``"same"``：在输入的四周填充，使得当 `stride` 为 ``1`` 时，输入和输出的shape一致。待填充的量由算子内部计算，若为偶数，则均匀地填充在四周，若为奇数，多余的填充量将补充在底部/右侧。如果设置了此模式， `padding` 必须为0。
          - ``"valid"``：不对输入进行填充，返回输出可能的最大高度和宽度，不能构成一个完整stride的额外的像素将被丢弃。如果设置了此模式， `padding` 必须为0。
          - ``"pad"``：对输入填充指定的量。在这种模式下，在输入的高度和宽度方向上填充的量由 `padding` 参数指定。如果设置此模式， `padding` 必须大于或等于0。

        - **padding** (Union[int, tuple[int]]，可选) - 输入的高度和宽度方向上填充的数量。数据类型为int或包含4个整数的tuple。如果 `padding` 是一个整数，那么上、下、左、右的填充都等于 `padding` 。如果 `padding` 是一个有4个整数的tuple，那么上、下、左、右的填充分别等于 `padding[0]` 、 `padding[1]` 、 `padding[2]` 和 `padding[3]` 。值应该要大于等于0，默认值： ``0`` 。
        - **dilation** (Union(int, tuple[int])，可选) - 卷积核膨胀尺寸。可以为单个int，或者由两个/四个int组成的tuple。单个int表示在高度和宽度方向的膨胀尺寸均为该值。两个int组成的tuple分别表示在高度和宽度方向的膨胀尺寸。若为四个int，N、C两维度int默认为1，H、W两维度分别对应高度和宽度上的膨胀尺寸。
          假设 :math:`dilation=(d0, d1)`, 则卷积核在高度方向间隔 `d0-1` 个元素进行采样，在宽度方向间隔 `d1-1` 个元素进行采样。高度和宽度上取值范围分别为[1, H]和[1, W]。默认值： ``1`` 。
        - **group** (int，可选) - 将过滤器拆分为组， `in_channels` 和 `out_channels` 必须可被 `group` 整除。如果组数等于 `in_channels` 和 `out_channels` ，这个二维卷积层也被称为二维深度卷积层。默认值： ``1`` 。
        - **has_bias** (bool，可选) - Conv2d层是否添加偏置参数。默认值： ``False`` 。
        - **weight_init** (Union[Tensor, str, Initializer, numbers.Number]，可选) - 权重参数的初始化方法。它可以是Tensor，str，Initializer或numbers.Number。当使用str时，可选 ``"TruncatedNormal"`` ， ``"Normal"`` ， ``"Uniform"`` ， ``"HeUniform"`` 和 ``"XavierUniform"`` 分布以及常量 ``"One"`` 和 ``"Zero"`` 分布的值，可接受别名 ``"xavier_uniform"`` ， ``"he_uniform"`` ， ``"ones"`` 和 ``"zeros"`` 。上述字符串大小写均可。更多细节请参考 `Initializer <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.common.initializer.html>`_, 的值。默认值： ``None`` ，权重使用 ``"HeUniform"`` 初始化。
        - **bias_init** (Union[Tensor, str, Initializer, numbers.Number]，可选) - 偏置参数的初始化方法。可以使用的初始化方法与 `weight_init` 相同。更多细节请参考 `Initializer <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.common.initializer.html>`_ 的值。默认值： ``None`` ，偏差使用 ``"Uniform"`` 初始化。
        - **data_format** (str，可选) - 数据格式的可选值有 ``"NHWC"`` ， ``"NCHW"`` 。默认值： ``"NCHW"`` 。
        - **dtype** (:class:`mindspore.dtype`) - Parameters的dtype。默认值： ``mstype.float32`` 。

    输入：
        - **x** (Tensor) - Shape为 :math:`(N, C_{in}, H_{in}, W_{in})` 或者 :math:`(N, H_{in}, W_{in}, C_{in})` 的Tensor。

    输出：
        Tensor，shape为 :math:`(N, C_{out}, H_{out}, W_{out})` 或者 :math:`(N, H_{out}, W_{out}, C_{out})` 。

        pad_mode为 ``"same"`` 时：

        .. math::
            \begin{array}{ll} \\
                H_{out} = \left \lceil{\frac{H_{in}}{\text{stride[0]}}} \right \rceil \\
                W_{out} = \left \lceil{\frac{W_{in}}{\text{stride[1]}}} \right \rceil \\
            \end{array}

        pad_mode为 ``"valid"`` 时：

        .. math::
            \begin{array}{ll} \\
                H_{out} = \left \lceil{\frac{H_{in} - \text{dilation[0]} \times (\text{kernel_size[0]} - 1) }
                {\text{stride[0]}}} \right \rceil \\
                W_{out} = \left \lceil{\frac{W_{in} - \text{dilation[1]} \times (\text{kernel_size[1]} - 1) }
                {\text{stride[1]}}} \right \rceil \\
            \end{array}

        pad_mode为 ``"pad"`` 时：

        .. math::
            \begin{array}{ll} \\
                H_{out} = \left \lfloor{\frac{H_{in} + padding[0] + padding[1] - (\text{kernel_size[0]} - 1) \times
                \text{dilation[0]} - 1 }{\text{stride[0]}} + 1} \right \rfloor \\
                W_{out} = \left \lfloor{\frac{W_{in} + padding[2] + padding[3] - (\text{kernel_size[1]} - 1) \times
                \text{dilation[1]} - 1 }{\text{stride[1]}} + 1} \right \rfloor \\
            \end{array}

    异常：
        - **TypeError** - 如果 `in_channels` ， `out_channels` 或者 `group` 不是整数。
        - **TypeError** - 如果 `kernel_size` ， `stride`，`padding` 或者 `dilation` 既不是整数也不是tuple。
        - **ValueError** - 如果 `in_channels` ， `out_channels`，`kernel_size` ， `stride` 或者 `dilation` 小于1。
        - **ValueError** - 如果 `padding` 小于0。
        - **ValueError** - 如果 `pad_mode` 不是 ``"same"`` ， ``"valid"`` 或 ``"pad"`` 。
        - **ValueError** - 如果 `padding` 是一个长度不等于4的tuple。
        - **ValueError** - 如果 `pad_mode` 不等于"pad"且 `padding` 不等于(0,0,0,0)。
        - **ValueError** - 如果 `data_format` 既不是"NCHW"也不是"NHWC"。
