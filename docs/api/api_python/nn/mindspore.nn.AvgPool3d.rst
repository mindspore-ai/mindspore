mindspore.nn.AvgPool3d
=======================

.. py:class:: mindspore.nn.AvgPool3d(kernel_size=1, strides=1, pad_mode="valid", pad=0, ceil_mode=False, count_include_pad=True, divisor_override=0)

    对输入的多维数据进行三维的平均池化运算。

    在一个输入Tensor上应用3D max pooling，可被视为组成一个3D平面。

    通常，输入的shape为 :math:`(N_{in}, C_{in}, D_{in}, H_{in}, W_{in})` ，AvgPool3D输出 :math:`(D_{in}, H_{in}, W_{in})` 区域平均值。给定 `kernel_size` 为 :math:`ks = (d_{ker}, h_{ker}, w_{ker})` 和 `strides` 为 :math:`s = (s_0, s_1, s_2)`，公式如下。

    .. warning::
        "kernel_size" 在[1, 255]的范围内取值。"strides" 在[1, 63]的范围内取值.

    .. math::
        \text{output}(N_i, C_j, d, h, w) =
        \frac{1}{d_{ker} * h_{ker} * w_{ker}} \sum_{l=0}^{d_{ker}-1} \sum_{m=0}^{h_{ker}-1} \sum_{n=0}^{w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times d + l, s_1 \times h + m, s_2 \times w + n)

    参数：
        - **kernel_size** (Union[int, tuple[int]]) - 指定池化核尺寸大小。如果为int，则同时代表池化核的深度，高度和宽度。如果为tuple，其值必须包含三个int，分别表示池化核的深度，高度和宽度。默认值：1。
        - **strides** (Union[int, tuple[int]]) - 池化操作的移动步长。如果为int，则同时代表池化核的深度，高度和宽度方向上的移动步长。如果为tuple，其值必须包含三个整数值，分别表示池化核的深度，高度和宽度方向上的移动步长。默认值：1。
        - **pad_mode** (str) - 指定池化填充模式，取值为"same"、"valid"或"pad"，不区分大小写。默认值："valid"。

          - **same** - 使用完成的方式进行填充。输出的深度，高度和宽度与输入相同。填充的总数将按深度、水平和垂直方向进行计算，并尽可能均匀的分布在头部和尾部、顶部和底部、左侧和右侧，此外，最后一个剩余的填充将填充在尾部、底部和右侧。如果设置了此模式，则"pad"必须为0。
          - **valid** - 使用丢弃的方式完成填充。输出的最大深度、高度和宽度将不带填充返回。不满足计算的多余像素会被丢弃。如果设置了此模式，则`pad`必须为0。
          - **pad** - 在深度，高度和宽度两侧进行隐式填充，`pad` 的值将会被填充到输入Tensor的边界，`pad` 必须大于或等于0。

        - **pad** (Union(int, tuple[int])) - 需要填充的pad值，默认值：0。如果 `pad` 为整数，则分别在头，尾，上，下，左，右都填充pad，如果`pad`是一个六个整数的元组，则分别在头，尾，上，下，左，右填充pad[0]，pad[1]，pad[2]，pad[3]，pad[4]，pad[5]。
        - **ceil_mode** (Union[bool, None]) - 若为True，使用ceil来计算输出shape。若为False，使用floor来计算输出shape。默认值：False。
        - **count_include_pad** (bool) - 平均计算是否包括零填充。默认值：True。
        - **divisor_override** (int) - 如果被指定为非0参数，该参数将会在平均计算中被用作除数，否则将会使用 `kernel_size` 作为除数，默认值：0。

    输入：
        - **x** (Tensor) - shape为 :math:`(N, C, D_{in}, H_{in}, W_{in})` 的Tensor。数据类型必须为float16或者float32。

    输出：
        shape为 :math:`(N, C, D_{out}, H_{out}, W_{out})` 的Tensor。数据类型与 `x` 一致。

    异常：
        - **TypeError** - `kernel_size` , `strides` 或 `pad` 既不是整数也不是元组。
        - **TypeError** - `ceil_mode` 或 `count_include_pad` 不是布尔类型。
        - **TypeError** - `pad_mode` 不是字符串。
        - **TypeError** - `divisor_override` 不是整数。
        - **ValueError** - `kernel_size` 或者 `strides` 中的数字不是正数。
        - **ValueError** - `kernel_size` 或 `strides` 是一个长度不为3的tuple。
        - **ValueError** - `pad_mode` 既不是'same'，也不是'valid'或者'pad'。
        - **ValueError** - `pad` 是一个长度不为6的tuple。
        - **ValueError** - `pad` 的元素小于0。
        - **ValueError** - `pad_mode` 不等于 'pad' 并且 `pad` 不等于0或者(0, 0, 0, 0, 0, 0)。
        - **ValueError** - `x` 的shape长度不等于5。
