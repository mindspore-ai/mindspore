mindspore.ops.Conv2D
====================

.. py:class:: mindspore.ops.Conv2D(out_channel, kernel_size, mode=1, pad_mode="valid", pad=0, stride=1, dilation=1, group=1, data_format="NCHW")

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
        - **out_channel** (int) - 指定输出通道数：:math:`C_{out}` 。
        - **kernel_size** (Union[int, tuple[int]]) - 指定二维卷积核的高度和宽度。可以为单个int或一个包含2个int组成的元组。单个整数表示该值同时适用于内核的高度和宽度。包含2个整数的元组表示第一个值用于高度，另一个值用于内核的宽度。
        - **mode** (int，可选) - 指定不同的卷积模式。此值目前未被使用。默认值： ``1`` 。
        - **pad_mode** (str，可选) - 指定填充模式，填充值为0。可选值为 ``"same"`` ， ``"valid"`` 或 ``"pad"`` 。默认值： ``"valid"`` 。

          - ``"same"``：在输入的四周填充，使得当 `stride` 为 ``1`` 时，输入和输出的shape一致。待填充的量由算子内部计算，若为偶数，则均匀地填充在四周，若为奇数，多余的填充量将补充在底部/右侧。如果设置了此模式， `pad` 必须为0。
          - ``"valid"``：不对输入进行填充，返回输出可能的最大高度和宽度，不能构成一个完整stride的额外的像素将被丢弃。如果设置了此模式， `pad` 必须为0。
          - ``"pad"``：对输入填充指定的量。在这种模式下，在输入的高度和宽度方向上填充的量由 `pad` 参数指定。如果设置此模式， `pad` 必须大于或等于0。

        - **pad** (Union(int, tuple[int])，可选) - 指当 `pad_mode` 为 ``"pad"`` 时，指定在输入 `x` 的高度和宽度方向上填充的数量。可以为单个int或包含四个int组成的tuple。如果 `pad` 是单个int，那么上、下、左、右的填充量都等于 `pad` 。如果 `pad` 是一个由四个int组成的tuple，那么上、下、左、右的填充分别等于 `pad[0]` 、 `pad[1]` 、 `pad[2]` 和 `pad[3]` 。int值应该要大于或等于0，默认值： ``0`` 。
        - **stride** (Union(int, tuple[int])，可选) - 卷积核移动的步长。可以为单个int，或由两个/四个int组成的tuple。单个int表示在高度和宽度方向的移动步长均为该值。两个int组成的tuple分别表示在高度和宽度方向的移动步长。若为四个int，N、C两维度默认为1，H、W两维度分别对应高度和宽度上的步长。默认值： ``1`` 。
        - **dilation** (Union(int, tuple[int])，可选) - 卷积核膨胀尺寸。可以为单个int，或者由两个/四个int组成的tuple。单个int表示在高度和宽度方向的膨胀尺寸均为该值。两个int组成的tuple分别表示在高度和宽度方向的膨胀尺寸。若为四个int，N、C两维度int默认为1，H、W两维度分别对应高度和宽度上的膨胀尺寸。
          假设 :math:`dilation=(d0, d1)`, 则卷积核在高度方向间隔 `d0-1` 个元素进行采样，在宽度方向间隔 `d1-1` 个元素进行采样。高度和宽度上取值范围分别为[1, H]和[1, W]。默认值： ``1`` 。
        - **group** (int，可选) - 分组卷积时在通道上分割输入 `x` 的组数。默认值： ``1`` 。
        - **data_format** (str，可选) - 数据格式的可选值有 ``"NHWC"`` ， ``"NCHW"`` 。默认值： ``"NCHW"`` 。

    输入：
        - **x** (Tensor) - 输入Tensor，shape为 :math:`(N, C_{in}, H_{in}, W_{in})` 或者 :math:`(N, H_{in}, W_{in}, C_{in}, )` ，具体哪种取决于 `data_format` 。
        - **weight** (Tensor) - 卷积核的值，其shape应为 :math:`(C_{out}, C_{in} / \text{group}, \text{kernel_size[0]}, \text{kernel_size[1]})` 。

    输出：
        Tensor，卷积操作后的值。shape为 :math:`(N, C_{out}, H_{out}, W_{out})` 或 :math:`(N, H_{out}, W_{out}, C_{out}, )` 。
        要了解不同的填充模式如何影响输出shape，请参考 :class:`mindspore.nn.Conv2d` 以获取更多详细信息。

    异常：
        - **TypeError** - `kernel_size` 、 `stride` 、 `pad` 或 `dilation` 既不是int也不是tuple。
        - **TypeError** - `out_channel` 或 `group` 不是int。
        - **ValueError** - `kernel_size` 、 `stride` 或 `diation` 小于1。
        - **ValueError** - `pad_mode` 不是"same"、"valid"或"pad"。
        - **ValueError** - `pad` 是一个长度不等于4的tuple。
        - **ValueError** - `pad_mode` 不等于"pad"，`pad` 不等于(0, 0, 0, 0)。
        - **ValueError** - `data_format` 既不是 ``"NCHW"`` ，也不是 ``"NHWC"`` 。
