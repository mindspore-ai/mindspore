mindspore.nn.AvgPool3d
=======================

.. py:class:: mindspore.nn.AvgPool3d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)

    对输入的多维数据进行三维的平均池化运算。

    在一个输入Tensor上应用3D max pooling，可被视为组成一个3D平面。

    通常，输入的shape为 :math:`(N_{in}, C_{in}, D_{in}, H_{in}, W_{in})` ，AvgPool3D输出 :math:`(D_{in}, H_{in}, W_{in})` 区域平均值。给定 `kernel_size` 为 :math:`ks = (d_{ker}, h_{ker}, w_{ker})` 和 `stride` 为 :math:`s = (s_0, s_1, s_2)`，公式如下。

    .. warning::
        "kernel_size" 在[1, 255]的范围内取值。"stride" 在[1, 63]的范围内取值。

    .. math::
        \text{output}(N_i, C_j, d, h, w) =
        \frac{1}{d_{ker} * h_{ker} * w_{ker}} \sum_{l=0}^{d_{ker}-1} \sum_{m=0}^{h_{ker}-1} \sum_{n=0}^{w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times d + l, s_1 \times h + m, s_2 \times w + n)

    参数：
        - **kernel_size** (Union[int, tuple[int]]) - 指定池化核尺寸大小。如果为int，则同时代表池化核的深度，高度和宽度。如果为tuple，其值必须包含三个int，分别表示池化核的深度，高度和宽度。
        - **stride** (Union[int, tuple[int]]) - 池化操作的移动步长。如果为int，则同时代表池化核的深度，高度和宽度方向上的移动步长。如果为tuple，其值必须包含三个整数值，分别表示池化核的深度，高度和宽度方向上的移动步长。默认值：None。
        - **padding** (Union(int, tuple[int])) - 需要填充的pad值，默认值：0。如果 `padding` 为整数，则分别在头，尾，上，下，左，右都填充padding，如果 `padding` 是一个六个整数的元组，则分别在头，尾，上，下，左，右填充padding[0]，padding[1]，padding[2]，padding[3]，padding[4]，padding[5]。
        - **ceil_mode** (bool) - 若为True，使用ceil来计算输出shape。若为False，使用floor来计算输出shape。默认值：False。
        - **count_include_pad** (bool) - 平均计算是否包括零填充。默认值：True。
        - **divisor_override** (int) - 如果被指定为非0参数，该参数将会在平均计算中被用作除数，否则将会使用 `kernel_size` 作为除数，默认值：None。

    输入：
        - **x** (Tensor) - shape为 :math:`(N, C, D_{in}, H_{in}, W_{in})` 或者 :math:`(C, D_{in}, H_{in}, W_{in})` 的Tensor。数据类型必须为float16或者float32。

    输出：
        shape为 :math:`(N, C, D_{out}, H_{out}, W_{out})` 或者 :math:`(C, D_{in}, H_{in}, W_{in})` 的Tensor。数据类型与 `x` 一致。

    异常：
        - **TypeError** - `kernel_size` ，`stride` 或 `padding` 既不是整数也不是元组。
        - **TypeError** - `ceil_mode` 或 `count_include_pad` 不是布尔类型。
        - **TypeError** - `data_format` 不是字符串。
        - **TypeError** - `divisor_override` 不是整数。
        - **ValueError** - `kernel_size` 或者 `stride` 中的数字不是正数。
        - **ValueError** - `kernel_size` 或 `stride` 是一个长度不为3的tuple。
        - **ValueError** - `padding` 是一个长度不为6的tuple。
        - **ValueError** - `padding` 的元素小于0。
        - **ValueError** - `x` 的shape长度不等于5。
