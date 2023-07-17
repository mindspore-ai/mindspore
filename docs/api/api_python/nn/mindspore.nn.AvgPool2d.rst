mindspore.nn.AvgPool2d
=======================

.. py:class:: mindspore.nn.AvgPool2d(kernel_size=1, stride=1, pad_mode='valid', padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None, data_format='NCHW')

    在输入Tensor上应用2D平均池化运算，可视为二维输入平面的组合。

    通常，输入的shape为 :math:`(N_{in}, C_{in}, H_{in}, W_{in})` ，AvgPool2d的输出为 :math:`(H_{in}, W_{in})` 维度的区域平均值。给定 `kernel_size` 为 :math:`ks = (h_{ker}, w_{ker})` 和 `stride` :math:`s = (s_0, s_1)`，公式定义如下：

    .. math::
        \text{output}(N_i, C_j, h, w) = \frac{1}{h_{ker} * w_{ker}} \sum_{m=0}^{h_{ker}-1} \sum_{n=0}^{w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times h + m, s_1 \times w + n)

    参数：
        - **kernel_size** (Union[int, tuple[int]]) - 指定池化核尺寸大小。如果为整数或单元素tuple，则代表池化核的高和宽。如果为tuple且长度不为 ``1`` ，其值必须包含两个整数值分别表示池化核的高和宽。默认值： ``1`` 。
        - **stride** (Union[int, tuple[int]]) - 池化操作的移动步长，如果为整数或单元素tuple，则代表池化核的高和宽方向的移动步长。如果为tuple且长度不为 ``1`` ，其值必须包含两个整数值分别表示池化核的高和宽的移动步长。默认值： ``1`` 。
        - **pad_mode** (str，可选) - 指定填充模式，填充值为0。可选值为 ``"same"`` ， ``"valid"`` 或 ``"pad"`` 。默认值： ``"valid"`` 。

          - ``"same"``：在输入的四周填充，使得当 `stride` 为 ``1`` 时，输入和输出的shape一致。待填充的量由算子内部计算，若为偶数，则均匀地填充在四周，若为奇数，多余的填充量将补充在底部/右侧。如果设置了此模式， `padding` 必须为0。
          - ``"valid"``：不对输入进行填充，返回输出可能的最大高度和宽度，不能构成一个完整stride的额外的像素将被丢弃。如果设置了此模式， `padding` 必须为0。
          - ``"pad"``：对输入填充指定的量。在这种模式下，在输入的高度和宽度方向上填充的量由 `padding` 参数指定。如果设置此模式， `padding` 必须大于或等于0。

        - **padding** (Union(int, tuple[int], list[int])) - 池化填充值，只有 `pad` 模式才能设置为非 ``0`` 。默认值： ``0`` 。 `padding` 只能是一个整数或者包含一个或两个整数的元组，若 `padding` 为一个整数或者包含一个整数的tuple/list，则会分别在输入的上下左右四个方向进行 `padding` 次的填充，若 `padding` 为一个包含两个整数的tuple/list，则会在输入的上下进行 `padding[0]` 次的填充，在输入的左右进行 `padding[1]` 次的填充。
        - **ceil_mode** (bool) - 若为 ``True`` ，使用ceil来计算输出shape。若为 ``False`` ，使用floor来计算输出shape。默认值： ``False`` 。
        - **count_include_pad** (bool) - 平均计算是否包括零填充。默认值： ``True`` 。
        - **divisor_override** (int) - 如果被指定为非0参数，该参数将会在平均计算中被用作除数，否则将会使用 `kernel_size` 作为除数，默认值： ``None`` 。
        - **data_format** (str) - 输入数据格式可为 ``'NHWC'`` 或 ``'NCHW'`` 。默认值： ``'NCHW'`` 。

    输入：
        - **x** (Tensor) - 输入数据的shape为 :math:`(N, C_{in}, H_{in}, W_{in})` 或 :math:`(C_{in}, H_{in}, W_{in})` 的Tensor。

    输出：
        输出数据的shape为 :math:`(N, C_{out}, H_{out}, W_{out})` 或 :math:`(C_{out}, H_{out}, W_{out})` 的Tensor。

        其中，如果 `pad_mode` 为 `pad` 模式时，输出的shape计算公式如下：

        .. math::
            H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] -
            \text{kernel_size}[0]}{\text{stride}[0]} + 1\right\rfloor

        .. math::
            W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] -
            \text{kernel_size}[1]}{\text{stride}[1]} + 1\right\rfloor

    异常：
        - **TypeError** - `kernel_size` 或 `strides` 既不是整数也不是元组。
        - **ValueError** - `pad_mode` 既不是"valid"，也不是"same" 或者 "pad"，不区分大小写。
        - **ValueError** - `data_format` 既不是'NCHW'，也不是'NHWC'。
        - **ValueError** - `data_format` 为 'NHWC' 时，使用了 `padding` 或者 `ceil_mode` 或者 `count_include_pad` 或者 `divisor_override` 或者 `pad_mode` 为 `pad`。
        - **ValueError** - `kernel_size` 或 `stride` 小于1。
        - **ValueError** - `padding` 为tuple/list时长度不为1或2。
        - **ValueError** - `x` 的shape长度不等于3或4。
        - **ValueError** - `divisor_override` 小于等于0。
        - **ValueError** -  `pad_mode` 不为 "pad" 的时候 `padding` 为非0。
