mindspore.ops.conv3d
====================

.. py:function:: mindspore.ops.conv3d(input, weight, bias=None, stride=1, pad_mode="valid", padding=0, dilation=1, groups=1)

    对输入Tensor计算三维卷积。通常，输入Tensor的shape为 :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` ，其中 :math:`N` 为batch size，:math:`C` 为通道数，:math:`D, H, W` 分别为特征图的深度、高度和宽度。

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

    卷积核shape为 :math:`(\text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})` ，其中 :math:`\text{kernel_size[0]}` 、
    :math:`\text{kernel_size[1]}` 和 :math:`\text{kernel_size[2]}` 分别是卷积核的深度、高度和宽度。若考虑到输入输出通道以及 `group` ，则完整卷积核的shape为
    :math:`(C_{out}, C_{in} / \text{group}, \text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})` ，
    其中 `group` 是分组卷积时在通道上分割输入 `x` 的组数。

    想更深入了解卷积层，请参考论文 `Gradient Based Learning Applied to Document Recognition <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_ 。

    .. note::

        1. 在Ascend平台上，目前只支持 :math:`groups=1` 。
        2. 在Ascend平台上，目前只支持 :math:`dilation=1` 。

    参数：
        - **input** (Tensor) - shape为 :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` 的Tensor。
        - **weight** (Tensor) - shape为 :math:`(C_{out}, C_{in} / \text{groups}, \text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})`  ，则卷积核的大小为 :math:`(\text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})` 。
        - **bias** (Tensor，可选) - 偏置Tensor，shape为 :math:`(C_{out})` 的Tensor。如果 `bias` 是None，将不会添加偏置。默认值： ``None`` 。
        - **stride** (Union[int, tuple[int]]，可选) - 卷积核移动的步长，可以为单个int或三个int组成的tuple。一个int表示在深度、高度和宽度方向的移动步长均为该值。三个int组成的tuple分别表示在深度、高度和宽度方向的移动步长。默认值： ``1`` 。
        - **pad_mode** (str，可选) - 指定填充模式。取值为 ``"same"`` ， ``"valid"`` ，或 ``"pad"`` 。默认值： ``"valid"`` 。

          - ``"same"``：输出的高度和宽度分别与输入整除 `stride` 后的值相同。填充将被均匀地添加到高和宽的两侧，剩余填充量将被添加到维度末端。若设置该模式，`padding` 的值必须为0。
          - ``"valid"``：在不填充的前提下返回有效计算所得的输出。不满足计算的多余像素会被丢弃。如果设置此模式，则 `padding` 的值必须为0。
          - ``"pad"``：对输入 `input` 进行填充。在输入的高度和宽度方向上填充 `padding` 大小的0。如果设置此模式， `padding` 必须大于或等于0。
        
        - **padding** (Union[int, tuple[int], list[int]]，可选) - 输入 `input` 的深度、高度和宽度方向上填充的数量。数据类型为int或包含3个int组成的tuple。如果 `padding` 是一个int，那么前、后、上、下、左、右的填充都等于 `padding` 。如果 `padding` 是一个有3个int组成的tuple，那么前、后的填充为 `padding[0]` ，上、下的填充为 `padding[1]` ，左、右的填充为 `padding[2]` 。值必须大于等于0，默认值： ``0`` 。
        - **dilation** (Union[int, tuple[int]]，可选) - 卷积核膨胀尺寸。数据类型为int或由3个int组成的tuple： :math:`(dilation_d, dilation_h, dilation_w)`。目前在Ascend后端， 只支持该值为1。若 :math:`k > 1` ，则卷积核间隔 `k` 个元素进行采样。前后、垂直和水平方向上，其取值范围分别为[1, D]、[1, H]和[1, W]。默认值： ``1`` 。
        - **groups** (int，可选) - 将过滤器拆分的组数， `in_channels` 和 `out_channels` 必须可被 `group` 整除。默认值： ``1`` 。

    返回：
        Tensor，卷积后的值。shape为 :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` 。

        `pad_mode` 为 ``"same"`` 时：

        .. math::
            \begin{array}{ll} \\
                D_{out} = \left \lceil{\frac{D_{in}}{\text{stride[0]}}} \right \rceil \\
                H_{out} = \left \lceil{\frac{H_{in}}{\text{stride[1]}}} \right \rceil \\
                W_{out} = \left \lceil{\frac{W_{in}}{\text{stride[2]}}} \right \rceil \\
            \end{array}

        `pad_mode` 为 ``"valid"`` 时：

        .. math::
            \begin{array}{ll} \\
                D_{out} = \left \lfloor{\frac{D_{in} - \text{dilation[0]} \times (\text{kernel_size[0]} - 1) }
                {\text{stride[0]}} + 1} \right \rfloor \\
                H_{out} = \left \lfloor{\frac{H_{in} - \text{dilation[1]} \times (\text{kernel_size[1]} - 1) }
                {\text{stride[1]}} + 1} \right \rfloor \\
                W_{out} = \left \lfloor{\frac{W_{in} - \text{dilation[2]} \times (\text{kernel_size[2]} - 1) }
                {\text{stride[2]}} + 1} \right \rfloor \\
            \end{array}

        `pad_mode` 为 ``"pad"`` 时：

        .. math::
            \begin{array}{ll} \\
                D_{out} = \left \lfloor{\frac{D_{in} + padding[0] + padding[0] - (\text{dilation[0]} - 1) \times
                \text{kernel_size[0]} - 1 }{\text{stride[0]}} + 1} \right \rfloor \\
                H_{out} = \left \lfloor{\frac{H_{in} + padding[1] + padding[1] - (\text{dilation[1]} - 1) \times
                \text{kernel_size[1]} - 1 }{\text{stride[1]}} + 1} \right \rfloor \\
                W_{out} = \left \lfloor{\frac{W_{in} + padding[2] + padding[2] - (\text{dilation[2]} - 1) \times
                \text{kernel_size[2]} - 1 }{\text{stride[2]}} + 1} \right \rfloor \\
            \end{array}

    异常：
        - **TypeError** -  `stride` 、 `padding` 或 `dilation` 既不是int也不是tuple。
        - **TypeError** -  `groups` 不是int。
        - **TypeError** -  `bias` 不是Tensor。
        - **ValueError** - `bias` 的shape不是 :math:`(C_{out})` 。
        - **ValueError** - `stride` 或 `diation` 小于1。
        - **ValueError** - `pad_mode` 不是"same"、"valid"或"pad"。
        - **ValueError** - `padding` 是一个长度不等于3的tuple。
        - **ValueError** - `pad_mode` 不等于"pad"时，`padding` 大于0。
