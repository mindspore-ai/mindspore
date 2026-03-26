mindspore.mint.nn.functional.conv_transpose2d
=============================================

.. py:function:: mindspore.mint.nn.functional.conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1)

    将2D转置卷积运算应用于由多个输入平面组成的输入图像，有时也称为反卷积（尽管它不是实际的反卷积）。

    更多参考详见 :class:`mindspore.mint.nn.ConvTranspose2d`。

    .. warning::
        - 这是一个实验性API，后续可能修改或删除。
        - 在输入非连续场景下， `output_padding` 必须小于 `stride` 。
        - 

    参数：
        - **input** (Tensor) - 输入Tensor，shape为 :math:`(minibatch, in\_channels, iH, iW)` 或 :math:`(in\_channels, iH, iW)` 。
        - **weight** (Tensor) - 卷积核，shape为 :math:`(in\_channels, \frac{out\_channels}{\text{groups}}, kH, kW)` 。
        - **bias** (Tensor, 可选) - 偏置，shape为 :math:`(out\_channels)` 。默认值： ``None`` 。
        - **stride** (Union[int, tuple(int), list[int]], 可选) - 卷积的步长。可以为1个整数、1个元组 :math:`(sH, sW)` 或者1个列表 :math:`[sH, sW]` 。默认值： ``1`` 。
        - **padding** (Union[int, tuple(int), list[int]], 可选) - :math:`dilation * (kernel\_size - 1) - padding` 零填充将添加到输入中每个维度的两侧。可以为1个整数、1个元组 :math:`(padH, padW)` 或者1个列表 :math:`[padH, padW]` 。默认值： ``0`` 。
        - **output_padding** (Union[int, tuple(int), list[int]], 可选) - 在输出形状中每个维度的一侧增加额外的尺寸。可以为1个整数、1个元组 :math:`(out\_padH, out\_padW)` 或者1个列表 :math:`[out\_padH, out\_padW]` 。 `output_padding` 的值必须小于 `stride` 或 `dilation` 。默认值： ``0`` 。
        - **groups** (int, 可选) - 将输入分成 `groups` 组。:math:`in\_channels` 应能被 `groups` 整除。默认值： ``1`` 。
        - **dilation** (Union[int, tuple(int), list[int]], 可选) - 内核元素之间的间距。可以为1个整数、1个元组 :math:`(dH, dW)` 或者1个列表 :math:`[dH, dW]` 。默认值： ``1`` 。

    返回：
        Tensor， shape为 :math:`(minibatch, out\_channels, oH, oW)` 或 :math:`(out\_channels, oH, oW)` 。其中：

        .. math::
            oH = (iH - 1) \times sH - 2 \times padH + dH \times (kH - 1) + out\_padH + 1
        .. math::
            oW = (iW - 1) \times sW - 2 \times padW + dW \times (kW - 1) + out\_padW + 1

    异常：
        - **TypeError** - `stride`， `padding`， `output_padding` 或 `dilation` 既不是int，也不是tuple或list。
        - **TypeError** - `groups` 不是int。
        - **ValueError** - `bias` 的shape不是 :math:`(out\_channels)` 。
        - **ValueError** - `stride` 或 `dilation` 小于1。
        - **ValueError** - `padding` 或 `output_padding` 小于0。
        - **ValueError** - `stride`， `padding`， `output_padding` 或 `dilation` 是tuple或list，且其长度不等于2。
