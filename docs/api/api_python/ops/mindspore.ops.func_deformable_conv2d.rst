mindspore.ops.deformable_conv2d
===============================

.. py:function:: mindspore.ops.deformable_conv2d(x, weight, offsets, kernel_size, strides, padding, bias=None, dilations=(1, 1, 1, 1), groups=1, deformable_groups=1, modulated=True)

    给定4D的Tensor输入 `x` ， `weight` 和 `offsets` ，计算一个2D的可变形卷积。可变形卷积运算可以表达如下：

    可变形卷积v1：

    .. math::
        y(p)=\sum_{k=1}^{K}w_{k}\cdot x(p+p_{k}+\Delta{p_{k}})

    可变形卷积v2：

    .. math::
        y(p)=\sum_{k=1}^{K}w_{k}\cdot x(p+p_{k}+\Delta{p_{k}})\cdot \Delta{m_{k}}

    其中 :math:`\Delta{p_{k}}` 和 :math:`\Delta{m_{k}}` 分别为第k个位置的可学习偏移和调制标量。细节请参考论文 `Deformable ConvNets v2: More Deformable, Better Results <https://arxiv.org/abs/1811.11168>`_ 和 `Deformable Convolutional Networks <https://arxiv.org/abs/1703.06211>`_ 。

    参数：
        - **x** (Tensor) - 一个四维Tensor，表示输入图像。数据格式为"NCHW"，shape为 :math:`(N, C_{in}, H_{in}, W_{in})` 。Dytpe为float16或float32。
        - **weight** (Tensor) - 一个四维Tensor，表示可学习的滤波器。数据类型必须与 `x` 相同，shape为 :math:`(C_{out}, C_{in} / groups, H_{f}, W_{f})` 。
        - **offsets** (Tensor) - 一个四维Tensor，存储x和y坐标的偏移，以及可变形卷积的输入掩码mask。数据格式为"NCHW"，shape为 :math:`(batch, 3 * deformable\_groups * H_{f} * W_{f}, H_{out}, W_{out})` ，注意其中C维度的存储顺序为(offset_x, offset_y, mask)。数据类型必须与 `x` 相同。
        - **kernel_size** (tuple[int]) - 一个包含两个整数的元组，表示卷积核的大小。
        - **strides** (tuple[int]) - 一个包含四个整数的元组，表示对于输入的每个维度的滑动窗口步长。其维度顺序依据 `x` 的数据格式，对应N和C维度的值必须设置成1。
        - **padding** (tuple[int]) - 一个包含四个整数的元组，表示沿（上，下，左，右）四个方向往输入填充的像素点个数。
        - **bias** (Tensor, 可选) - 一个一维Tensor，表示加到卷积输出的偏置参数。shape为 :math:`(C_{out})` 。默认值为None。
        - **dilations** (tuple[int], 可选) - 一个包含四个整数的元组，表示对于输入的每个维度的膨胀系数。其维度顺序依据 `x` 的数据格式，对应N和C维度的值必须设置成1。默认值为(1, 1, 1, 1)。
        - **groups** (int, 可选) - 一个int32类型的整数，表示从输入通道到输出通道的阻塞连接数。输入通道数和输出通道数必须都能被 `groups` 整除。默认值为1。
        - **deformable_groups** (int, 可选) - 一个int32类型的整数，表示可变形卷积组数。输入通道数必须能被 `deformable_groups` 整除。默认值为1。
        - **modulated** (bool, 可选) - 指定可变形二维卷积的版本。True表示v2，False表示v1。当前只支持设置为v2版本。默认值为True。

    返回：
        Tensor，一个四维Tensor，表示输出特征图。数据类型与 `x` 相同，数据格式为"NCHW"，shape为 :math:`(N, C_{out}, H_{out}, W_{out})` 。

        .. math::
            \begin{array}{ll} \\
                H_{out} = \left \lfloor{\frac{H_{in} + padding[0] + padding[1] - (H_{f} - 1) \times
                \text{dilations[2]} - 1 }{\text{stride[0]}} + 1} \right \rfloor \\
                W_{out} = \left \lfloor{\frac{W_{in} + padding[2] + padding[3] - (W_{f} - 1) \times
                \text{dilations[3]} - 1 }{\text{stride[1]}} + 1} \right \rfloor \\
            \end{array}

    异常：
        - **TypeError** - 如果 `strides`， `padding`， `kernel_size` 或者 `dilations` 不是一个整数元组。
        - **TypeError** - 如果 `modulated` 不是一个布尔值。
        - **ValueError** - 如果 `strides`， `padding`， `kernel_size` 或者 `dilations` 的元组不是期望的大小。
        - **ValueError** - 如果 `strides` 或者 `dilations` 对应N和C维度的值不为1。
        - **ValueError** - 如果 `modulated` 的值不是True。

    .. note::
        - 这是一个实验性质的接口，将来有可能被修改或删除。
