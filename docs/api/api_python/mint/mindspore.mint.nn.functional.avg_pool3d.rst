mindspore.mint.nn.functional.avg_pool3d
========================================

.. py:function:: mindspore.mint.nn.functional.avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)

    在输入Tensor上应用3d平均池化，输入Tensor可以看作是由一系列3d平面组成的。

    一般地，输入的shape为 :math:`(N, C, D_{in}, H_{in}, W_{in})` ，输出 :math:`(D_{in}, H_{in}, W_{in})` 维度的区域平均值。给定 `kernel_size` 为 :math:`(kD, kH, kW)` 和 `stride` ，运算如下：

    .. math::
        \text{output}(N_i, C_j, d, h, w) = \frac{1}{kD * kH * kW} \sum_{l=0}^{kD-1} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1}

        \text{input}(N_i, C_j, stride[0] \times d + l, stride[1] \times h + m, stride[2] \times w + n)

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - shape为 :math:`(N, C, D_{in}, H_{in}, W_{in})` 或 :math:`(C, D_{in}, H_{in}, W_{in})` 的Tensor。
        - **kernel_size** (Union[int, tuple[int], list[int]]) - 指定池化核尺寸大小，可以是单个整数、一个元组 :math:`(kD, kH, kW)` 或者一个列表 :math:`[kD, kH, kW]` 。
        - **stride** (Union[int, tuple[int], list[int]], 可选) - 池化操作的移动步长，可以是单个整数、一个元组 :math:`(sD, sH, sW)` 或者一个列表 :math:`[sD, sH, sW]` 。默认值： ``None``，此时其值等于 `kernel_size` 。
        - **padding** (Union[int, tuple[int], list[int]], 可选) - 池化填充长度，可以是单个整数、一个元组 :math:`(padD, padH, padW)` 或者一个列表 :math:`[padD, padH, padW]` 。默认值： ``0``。
        - **ceil_mode** (bool, 可选) - 如果为 ``True`` ，用ceil代替floor来计算输出的shape。默认值： ``False`` 。
        - **count_include_pad** (bool, 可选) - 如果为 ``True`` ，平均计算将包括零填充。默认值： ``True`` 。
        - **divisor_override** (int, 可选) - 如果指定了该值，它将在平均计算中用作除数，否则，将使用池化区域的大小。默认值： ``None``。

    返回：
        Tensor，其shape为 :math:`(N, C, D_{out}, H_{out}, W_{out})` 或 :math:`(C, D_{out}, H_{out}, W_{out})` 。

        .. math::
            \begin{array}{ll} \\
                D_{out} = \frac{D_{in} + 2 \times padding[0] - kernel\_size[0]}{stride[0]} + 1 \\
                H_{out} = \frac{H_{in} + 2 \times padding[1] - kernel\_size[1]}{stride[1]} + 1 \\
                W_{out} = \frac{W_{in} + 2 \times padding[2] - kernel\_size[2]}{stride[2]} + 1
            \end{array}

    异常：
        - **TypeError** - `input` 不是一个Tensor。
        - **TypeError** - `kernel_size` 或 `stride` 既不是int，也不是tuple或list。
        - **TypeError** - `ceil_mode` 或 `count_include_pad` 不是bool。
        - **TypeError** - `divisor_override` 不是int或 ``None``。
        - **ValueError** - `input` 的维度不等于4或5。
        - **ValueError** - `kernel_size` 或 `stride` 小于1。
        - **ValueError** - `padding` 的值小于0。
        - **ValueError** - `kernel_size`、 `padding` 或 `stride` 是tuple或list，且其长度不等于1或3。
