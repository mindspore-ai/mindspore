mindspore.ops.Conv3D
====================

.. py:class:: mindspore.ops.Conv3D(out_channel, kernel_size, mode=1, pad_mode="valid", pad=0, stride=1, dilation=1, group=1, data_format="NCDHW")

    三维卷积层。

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
        - **out_channel** (int) - 指定输出通道数：:math:`C_{out}` 。
        - **kernel_size** (Union[int, tuple[int]]) - 指定三维卷积核的高度和宽度。可以为单个int或一个包含3个int组成的元组。单个整数表示该值同时适用于内核的深度、高度和宽度。包含3个整数的元组表示第一个值用于深度，另两个值用于高度和宽度。
        - **mode** (int，可选) - 指定不同的卷积模式。此值目前未被使用。默认值： ``1`` 。
        - **stride** (Union[int, tuple[int]]，可选) - 卷积核移动的步长，可以为单个int或三个int组成的tuple。一个int表示在深度、高度和宽度方向的移动步长均为该值。三个int组成的tuple分别表示在深度、高度和宽度方向的移动步长。默认值： ``1`` 。
        - **pad_mode** (str，可选) - 指定填充模式，填充值为0。可选值为 ``"same"`` ， ``"valid"`` 或 ``"pad"`` 。默认值： ``"valid"`` 。

          - ``"same"``：在输入的深度、高度和宽度维度进行填充，使得当 `stride` 为 ``1`` 时，输入和输出的shape一致。待填充的量由算子内部计算，若为偶数，则均匀地填充在四周，若为奇数，多余的填充量将补充在前方/底部/右侧。如果设置了此模式， `pad` 必须为0。
          - ``"valid"``：不对输入进行填充，返回输出可能的最大深度、高度和宽度，不能构成一个完整stride的额外的像素将被丢弃。如果设置了此模式， `pad` 必须为0。
          - ``"pad"``：对输入填充指定的量。在这种模式下，在输入的深度、高度和宽度方向上填充的量由 `pad` 参数指定。如果设置此模式， `pad` 必须大于或等于0。

        - **pad** (Union(int, tuple[int])，可选) - 指当 `pad_mode` 为 ``"pad"`` 时，指定在输入 `x` 的深度、高度和宽度方向上填充的数量。可以为单个int或包含六个int组成的tuple。如果 `pad` 是单个int，那么前、后、上、下、左、右的填充量都等于 `pad` 。如果 `pad` 是一个由六个int组成的tuple，那么前、后、上、下、左、右的填充分别等于 `pad[0]` 、 `pad[1]` 、 `pad[2]` 、 `pad[3]` 、 `pad[4]` 和 `pad[5]` 。int值应该要大于或等于0，默认值： ``0`` 。
        - **dilation** (Union[int, tuple[int]]，可选) - 卷积核膨胀尺寸。可以为单个int，或者由三个int组成的tuple。单个int表示在深度、高度和宽度方向的膨胀尺寸均为该值。三个int组成的tuple分别表示在深度、高度和宽度方向的膨胀尺寸。
          假设 :math:`dilation=(d0, d1, d2)`, 则卷积核在深度方向间隔 `d0-1` 个元素进行采样，在高度方向间隔 `d1-1` 个元素进行采样，在高度方向间隔 `d2-1` 个元素进行采样。深度、高度和宽度上取值范围分别为[1, D]、[1, H]和[1, W]。默认值： ``1`` 。
        - **group** (int，可选) - 将过滤器拆分的组数， `in_channels` 和 `out_channels` 必须可被 `group` 整除。默认值： ``1`` 。
        - **data_format** (str，可选) - 支持的数据模式。目前仅支持 ``"NCDHW"`` 。

    输入：
        - **x** (Tensor) - shape为 :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` 的Tensor。目前数据类型仅支持float16和float32。
        - **weight** (Tensor) - 若kernel shape为 :math:`(k_d, K_h, K_w)` ，则weight shape应为 :math:`(C_{out}, C_{in}/groups, k_d, K_h, K_w)` 。目前数据类型仅支持float16和float32。
        - **bias** (Tensor) - shape为 :math:`C_{out}` 的Tensor。如果 `bias` 为 ``None`` ，将不会添加偏置。默认值： ``None`` 。

    输出：
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
                D_{out} = \left \lfloor{\frac{D_{in} + pad[0] + pad[1] - (\text{dilation[0]} - 1) \times
                \text{kernel_size[0]} - 1 }{\text{stride[0]}} + 1} \right \rfloor \\
                H_{out} = \left \lfloor{\frac{H_{in} + pad[2] + pad[3] - (\text{dilation[1]} - 1) \times
                \text{kernel_size[1]} - 1 }{\text{stride[1]}} + 1} \right \rfloor \\
                W_{out} = \left \lfloor{\frac{W_{in} + pad[4] + pad[5] - (\text{dilation[2]} - 1) \times
                \text{kernel_size[2]} - 1 }{\text{stride[2]}} + 1} \right \rfloor \\
            \end{array}

    异常：
        - **TypeError** - `out_channel` 或 `group` 不是int。
        - **TypeError** - `kernel_size` 、 `stride` 、 `pad` 或 `dilation` 既不是int也不是Tuple。
        - **ValueError** - `out_channel` 、 `kernel_size` 、 `stride` 或 `dilation` 小于1。
        - **ValueError** - `pad` 小于0。
        - **ValueError** - `pad_mode` 取值非"same"、"valid"或"pad"。
        - **ValueError** - `pad` 为长度不等于6的Tuple。
        - **ValueError** - `pad_mode` 未设定为"pad"且 `pad` 不等于(0, 0, 0, 0, 0, 0)。
        - **ValueError** - `data_format` 取值非"NCDHW"。
