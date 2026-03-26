mindspore.mint.nn.functional.conv3d
====================================

.. py:function:: mindspore.mint.nn.functional.conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor

    对输入Tensor计算三维卷积。通常，输入Tensor的shape为 :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` 或 :math:`(C_{in}, D_{in}, H_{in}, W_{in})`，其中 :math:`N` 为batch size，:math:`C` 为通道数，:math:`D, H, W` 分别为特征图的深度、高度和宽度。

    根据以下公式计算输出：

    .. math::

        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{ccor}({\text{weight}(C_{\text{out}_j}, k), \text{X}(N_i, k)})

    其中， :math:`bias` 为输出偏置，:math:`ccor` 为 `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_ 操作，
    :math:`weight` 为卷积核的值， :math:`X` 为输入的特征图。

    以下是索引的含义：

    - :math:`i` 对应batch数，其范围为 :math:`[0, N-1]` ，其中 :math:`N` 为输入batch。

    - :math:`j` 对应输出通道，其范围为 :math:`[0, C_{out}-1]` ，其中 :math:`C_{out}` 为输出通道数，该值也等于卷积核的个数。

    - :math:`k` 对应输入通道数，其范围为 :math:`[0, C_{in}-1]` ，其中 :math:`C_{in}` 为输入通道数，该值也等于卷积核的通道数。

    因此，上面的公式中， :math:`{bias}(C_{\text{out}_j})` 为第 :math:`j` 个输出通道的偏置， :math:`{weight}(C_{\text{out}_j}, k)` 表示第 :math:`j` 个\
    卷积核在第 :math:`k` 个输入通道的卷积核切片， :math:`{X}(N_i, k)` 为特征图第 :math:`i` 个batch第 :math:`k` 个输入通道的切片。

    卷积核shape为 :math:`(kd, kh, kw)` ，其中 :math:`kd` 、
    :math:`kh` 和 :math:`kw` 分别是卷积核的深度、高度和宽度。若考虑到输入输出通道以及 `groups` ，则完整卷积核的shape为
    :math:`(C_{out}, C_{in} / \text{groups}, kd, kh, kw)` ，
    其中 `groups` 是分组卷积时在通道上分割输入 `input` 的组数。

    想更深入了解卷积层，请参考论文 `Gradient Based Learning Applied to Document Recognition <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_ 。

    以下罗列参数的一些限制条件。

    - input -- conv3d的输入. 输入的每个维度大小必须在[1, int32_max]范围内。
    - weight -- shape为 :math:`(C_{out}, C_{in} / groups, kd, kh, kw)` 的滤波器。:math:`kh` 和 :math:`kw` 的值在 [1, 511] 范围内。其余值在 [1, int32_max] 范围内。
      并且 :math:`kh*kw*k0` 必须小于 65536（其中 k0 为 16。如果数据类型是 float32，则 k0 为 8）。
    - bias -- 形状为 :math:`(C_{out})` 的偏置张量。其形状必须与权重的第一维相等。
    - stride -- 内核移动的步幅。它可以是一个整数或一个元组（表示为 :math:`(stride_d, stride_h, stride_w)` ）。
      其中，stride_h 和 stride_w 的范围是 [1, 63]，stride_d 的范围是 [1, 255]。
    - padding -- 如果 padding 是一个整数，则其范围为 [0, 255]。
    - dilation -- 该值的范围是 [1, 255]。
    - groups -- 该值的范围是 [1, 65535]。
    - :math:`C_{in} \% \text{groups} == 0 \quad \text{and} \quad C_{out} \% \text{groups} == 0` 。
    - :math:`weight[1] == C_{in} / groups` 。
    - :math:`H_{in} + PadUp + PadDown >= (kh - 1) * DilationH + 1` 。
    - :math:`W_{in} + PadLeft + PadRight >= (kw - 1) * DilationW + 1` 。
    - :math:`D_{in} + PadFront + PadBack >= (kd - 1) * DilationD + 1` 。
    - :math:`H_{out} = (H_{in} + PadUp + PadDown - ((kh - 1) * DilationH + 1)) / StrideH + 1` 。
    - :math:`W_{out} = (W_{in} + PadLeft + PadRight - ((kw - 1) * DilationW + 1)) / StrideW + 1` 。
    - :math:`D_{out} = (D_{in} + PadFront + PadBack - ((kd - 1) * DilationD + 1)) / StrideD + 1` 。
    - :math:`(D_{in}+PadFront+PadBack - ((kd-1)*DilationD+1)) \% StrideD <= PadBack` 。
    - :math:`(H_{in}+PadUp+PadDown - ((kh-1)*Dilationh+1)) \% StrideH <= PadDown` 。
    - :math:`stride_d <= kernel_d` 。
    - :math:`PadUp < kh` 且 :math:`PadDown < kh` 。当 `padding` = ``'valid'`` 时， PadUp 和 PadDown 取值是0。 当 `padding` = ``'same'`` 时， 对于high维度的PadUp能通过
      :math:`floor(((H_{out}-1) * strideH + (kh - 1) * DilationH + 1 - H_{in}) / 2)` 计算得到。
      用类似的方法能计算得到关于depth和width维度的padding值。且depth和width维度也有相同的约束条件。
    - :math:`((kh - 1) * DilationH - PadUp)` 取值区间为[0, 255]。深度和宽度维度具有相同的约束。
    - 如果 `padding` 为 ``'same'``， `stride` 必须为 1。

    参数：
        - **input** (Tensor) - shape为 :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` 的Tensor。
        - **weight** (Tensor) - shape为 :math:`(C_{out}, C_{in} / \text{groups}, kd, kh, kw)` ，则卷积核的大小为 :math:`(kd, kh, kw)` 。
        - **bias** (Tensor，可选) - 偏置Tensor，shape为 :math:`(C_{out})` 的Tensor。如果 `bias` 是None，将不会添加偏置。默认值： ``None`` 。
        - **stride** (Union(int, tuple[int], list[int])，可选) - 卷积核移动的步长，可以为单个int或三个int组成的tuple/list。一个int表示在深度、高度和宽度方向的移动步长均为该值。三个int组成的tuple/list分别表示在深度、高度和宽度方向的移动步长。默认值： ``1`` 。
        - **padding** (Union(int, tuple[int], list[int], str)，可选) - 输入 `input` 两侧的隐式填充。可以是字符串、一个整数或包含3个整数的元组/列表。如果 `padding` 是一个字符串，则可选值为 `"same"` 、 `"valid"` 。

          - ``"same"``：采用完成方式。输出的高度和宽度将等于输入 `input` 除以步幅。填充将尽可能在顶部和底部、左侧和右侧均匀计算。否则，最后一个额外的填充将从底部和右侧计算。如果设置了此模式，则 `stride` 必须为 ``1``。
          - ``"valid"``：采用丢弃的方式。输出的可能最大高度和宽度将在没有填充的情况下返回。多余的像素将被丢弃。

          如果 `padding` 是一个整数，则top、bottom、left和right的padding是相同的，等于padding。
          如果 `padding` 是一个包含3个整数的元组/列表，则head、tail、top、bottom、left和right的填充分别等于pad[0]、pad[0]、pad[1]、pad[1]、pad[2]和pad[2]。默认值： ``0`` 。

        - **dilation** (Union[int, tuple[int], list[int]]，可选) - 控制内核点之间的空间。默认值： ``1`` 。
        - **groups** (int，可选) - 将 `input` 拆分的组数。默认值： ``1`` 。

    返回：
        Tensor，dtype与 `input` 相同，shape为 :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`
        或 :math:`(C_{out}, D_{out}, H_{out}, W_{out})` 。

    异常：
        - **TypeError** -  `stride` 、 `padding` 或 `dilation` 既不是int，也不是tuple或list。
        - **TypeError** -  `groups` 不是int。
        - **TypeError** -  `bias` 不是Tensor。
