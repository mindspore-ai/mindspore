mindspore.mint.nn.functional.avg_pool2d
========================================

.. py:function:: mindspore.mint.nn.functional.avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)

    在输入Tensor上应用2D平均池化，输入Tensor可以看作是由一系列2D平面组成的。

    一般地，输入的shape为 :math:`(N, C, H_{in}, W_{in})` ，输出 :math:`(H_{in}, W_{in})` 维度的区域平均值。给定 `kernel_size` 为 :math:`(kH, kW)` 和 `stride` ，运算如下：
    
    .. note::
        在Atlas平台上，计算输入时，精度会从float32降到float16。

    .. math::
        \text{output}(N_i, C_j, h, w) = \frac{1}{kH * kW} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1}
            \text{input}(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)

    参数：
        - **input** (Tensor) - shape为 :math:`(N, C, H_{in}, W_{in})` 或 :math:`(C, H_{in}, W_{in})` 的Tensor。
        - **kernel_size** (Union[int, tuple[int], list[int]]) - 指定池化核尺寸大小，可以是单个整数或一个元组 :math:`(kH, kW)` 。
        - **stride** (Union[int, tuple[int], list[int]], 可选) - 池化操作的移动步长，可以是单个整数或一个元组 :math:`(sH, sW)` 。默认值： ``None``，此时其值等于 `kernel_size` 。
        - **padding** (Union[int, tuple[int], list[int]], 可选) - 池化填充长度，可以是单个整数或一个元组 :math:`(padH, padW)`。默认值： ``0``。
        - **ceil_mode** (bool, 可选) - 如果为 ``True`` ，用ceil代替floor来计算输出的shape。默认值： ``False`` 。
        - **count_include_pad** (bool, 可选) - 如果为 ``True`` ，平均计算将包括零填充。默认值： ``True`` 。
        - **divisor_override** (int, 可选) - 如果指定了该值，它将在平均计算中用作除数，否则，将使用池化区域的大小。默认值： ``None``。

    返回：
        Tensor，其shape为 :math:`(N, C, H_{out}, W_{out})` 或 :math:`(C, H_{out}, W_{out})` 。

        .. math::
            \begin{array}{ll} \\
                H_{out} = \frac{H_{in} + 2 \times padding[0] - kernel\_size[0]}{stride[0]} + 1 \\
                W_{out} = \frac{W_{in} + 2 \times padding[1] - kernel\_size[1]}{stride[1]} + 1
            \end{array}

    异常：
        - **TypeError** - `input` 不是一个Tensor。
        - **TypeError** - `kernel_size` 或 `stride` 既不是int也不是tuple。
        - **TypeError** - `ceil_mode` 或 `count_include_pad` 不是bool。
        - **TypeError** - `divisor_override` 不是int或None。
        - **ValueError** - `input` 的维度不等于3或4。
        - **ValueError** - `kernel_size` 或 `stride` 小于1。
        - **ValueError** - `padding` 的值小于0。
        - **ValueError** - `kernel_size`、 `padding` 或 `stride` 是tuple且其长度不等于1或2。
