mindspore.nn.Conv3d
=============================

.. py:class:: mindspore.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, pad_mode='same', padding=0, dilation=1, group=1, has_bias=False, weight_init=None, bias_init=None, data_format='NCDHW', dtype=mstype.float32)

    三维卷积层。

    对输入Tensor计算三维卷积。通常，输入Tensor的shape为 :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` ，其中 :math:`N` 为batch size，:math:`C` 为通道数，:math:`D, H, W` 分别为特征图的深度、高度和宽度。

    根据以下公式计算输出：

    .. math::

        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{ccor}({\text{weight}(C_{\text{out}_j}, k), \text{X}(N_i, k)})

    其中， :math:`bias` 为输出偏置，:math:`ccor` 为 `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_ 操作， 
    :math:`weight` 为卷积核的值， :math:`X` 为输入的特征图。

    - :math:`i` 对应batch数，其范围为 :math:`[0, N-1]` ，其中 :math:`N` 为输入batch。

    - :math:`j` 对应输出通道，其范围为 :math:`[0, C_{out}-1]` ，其中 :math:`C_{out}` 为输出通道数，该值也等于卷积核的个数。

    - :math:`k` 对应输入通道数，其范围为 :math:`[0, C_{in}-1]`，其中 :math:`C_{in}` 为输入通道数，该值也等于卷积核的通道数。

    因此，上面的公式中， :math:`{bias}(C_{\text{out}_j})` 为第 :math:`j` 个输出通道的偏置， :math:`{weight}(C_{\text{out}_j}, k)` 表示第 :math:`j` 个
    卷积核在第 :math:`k` 个输入通道的卷积核切片， :math:`{X}(N_i, k)` 为特征图第 :math:`i` 个batch第 :math:`k` 个输入通道的切片。

    卷积核shape为 :math:`(\text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})` ，其中 :math:`\text{kernel_size[0]}` 、
    :math:`\text{kernel_size[1]}` 和 :math:`\text{kernel_size[2]}` 分别是卷积核的深度、高度和宽度。若考虑到输入输出通道以及 `group` ，则完整卷积核的shape为
    :math:`(C_{out}, C_{in} / \text{group}, \text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})` ，
    其中 `group` 是分组卷积时在通道上分割输入 `x` 的组数。

    想更深入了解卷积层，请参考论文 `Gradient Based Learning Applied to Document Recognition <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_ 。

    .. note::
        在Ascend平台上，目前只支持深度卷积场景下的分组卷积运算。也就是说，当 `group>1` 的场景下，必须要满足 `in\_channels` = `out\_channels` = `group` 的约束条件。

    参数：
        - **in_channels** (int) - Conv3d层输入Tensor的空间维度。
        - **out_channels** (int) - Conv3d层输出Tensor的空间维度。
        - **kernel_size** (Union[int, tuple[int]]) - 指定三维卷积核的高度和宽度。可以为单个int或一个包含3个int组成的元组。单个整数表示该值同时适用于内核的深度、高度和宽度。包含3个整数的元组表示第一个值用于深度，另两个值用于高度和宽度。
        - **stride** (Union[int, tuple[int]]，可选) - 三维卷积核的移动步长。数据类型为整型或三个整型的tuple。一个整数表示在深度、高度和宽度方向的移动步长均为该值。三个整数的tuple分别表示在深度、高度和宽度方向的移动步长。默认值： ``1`` 。
        - **pad_mode** (str，可选) - 指定填充模式，填充值为0。可选值为 ``"same"`` ， ``"valid"`` 或 ``"pad"`` 。默认值： ``"same"`` 。

          - ``"same"``：在输入的深度、高度和宽度维度进行填充，使得当 `stride` 为 ``1`` 时，输入和输出的shape一致。待填充的量由算子内部计算，若为偶数，则均匀地填充在四周，若为奇数，多余的填充量将补充在前方/底部/右侧。如果设置了此模式， `padding` 必须为0。
          - ``"valid"``：不对输入进行填充，返回输出可能的最大深度、高度和宽度，不能构成一个完整stride的额外的像素将被丢弃。如果设置了此模式， `padding` 必须为0。
          - ``"pad"``：对输入填充指定的量。在这种模式下，在输入的深度、高度和宽度方向上填充的量由 `padding` 参数指定。如果设置此模式， `padding` 必须大于或等于0。

        - **padding** (Union(int, tuple[int])，可选) - 输入的深度、高度和宽度方向上填充的数量。数据类型为int或包含6个整数的tuple。如果 `padding` 是一个整数，则前部、后部、顶部，底部，左边和右边的填充都等于 `padding` 。如果 `padding` 是6个整数的tuple，则前部、尾部、顶部、底部、左边和右边的填充分别等于填充padding[0]、padding[1]、padding[2]、padding[3]、padding[4]和padding[5]。值应该要大于等于0，默认值： ``0`` 。
        - **dilation** (Union[int, tuple[int]]，可选) - 卷积核膨胀尺寸。可以为单个int，或者由三个int组成的tuple。单个int表示在深度、高度和宽度方向的膨胀尺寸均为该值。三个int组成的tuple分别表示在深度、高度和宽度方向的膨胀尺寸。
          假设 :math:`dilation=(d0, d1, d2)`，则卷积核在深度方向间隔 :math:`d0-1` 个元素进行采样，在高度方向间隔 :math:`d1-1` 个元素进行采样，在高度方向间隔 :math:`d2-1` 个元素进行采样。深度、高度和宽度上取值范围分别为[1, D]、[1, H]和[1, W]。默认值： ``1`` 。
        - **group** (int，可选) - 将过滤器拆分为组， `in_channels` 和 `out_channels` 必须可被 `group` 整除。默认值： ``1`` 。
        - **has_bias** (bool，可选) - Conv3d层是否添加偏置参数。默认值： ``False`` 。
        - **weight_init** (Union[Tensor, str, Initializer, numbers.Number]，可选) - 权重参数的初始化方法。它可以是Tensor，str，Initializer或numbers.Number。当使用str时，可选 ``"TruncatedNormal"`` ， ``"Normal"`` ， ``"Uniform"`` ， ``"HeUniform"`` 和 ``"XavierUniform"`` 分布以及常量 ``"One"`` 和 ``"Zero"`` 分布的值，可接受别名 ``"xavier_uniform"`` ， ``"he_uniform"`` ， ``"ones"`` 和 ``"zeros"`` 。上述字符串大小写均可。更多细节请参考 `Initializer <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.common.initializer.html>`_, 的值。默认值： ``None`` ，权重使用 ``"HeUniform"`` 初始化。
        - **bias_init** (Union[Tensor, str, Initializer, numbers.Number]，可选) - 偏置参数的初始化方法。可以使用的初始化方法与 `weight_init` 相同。更多细节请参考 `Initializer <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.common.initializer.html>`_, 的值。默认值： ``None`` ，偏差使用 ``"Uniform"`` 初始化。
        - **data_format** (str，可选) - 数据格式的可选值。目前仅支持 ``'NCDHW'`` 。
        - **dtype** (:class:`mindspore.dtype`) - Parameters的dtype。默认值： ``mstype.float32`` 。

    输入：
        - **x** (Tensor) - shape为 :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` 的Tensor。目前，CPU和GPU平台上输入数据类型支持float16和float32，Ascend平台上输入数据类型只支持float16。

    输出：
        Tensor，shape为 :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` 。

        pad_mode为 ``"same"`` 时：

        .. math::
            \begin{array}{ll} \\
                D_{out} = \left \lceil{\frac{D_{in}}{\text{stride[0]}}} \right \rceil \\
                H_{out} = \left \lceil{\frac{H_{in}}{\text{stride[1]}}} \right \rceil \\
                W_{out} = \left \lceil{\frac{W_{in}}{\text{stride[2]}}} \right \rceil \\
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
        - **TypeError** - `kernel_size` 、 `stride` 、 `padding` 或 `dilation` 既不是int也不是tuple。
        - **ValueError** - `out_channels` 、 `kernel_size` 、 `stride` 或 `dilation` 小于1。
        - **ValueError** - `padding` 小于0。
        - **ValueError** - `pad_mode` 不是 ``"same"`` ， ``"valid"`` 或 ``"pad"`` 。
        - **ValueError** - `padding` 是长度不等于6的tuple。
        - **ValueError** - `pad_mode` 不等于"pad"且 `padding` 不等于(0, 0, 0, 0, 0, 0)。
        - **ValueError** - `data_format` 不是"NCDHW"。
