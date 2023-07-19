mindspore.ops.AvgPool3D
========================

.. py:class:: mindspore.ops.AvgPool3D(kernel_size=1, strides=1, pad_mode="valid", pad=0, ceil_mode=False, count_include_pad=True, divisor_override=0, data_format="NCDHW")

    对输入的多维数据进行三维的平均池化运算。

    一般，输入shape为 :math:`(N, C, D_{in}, H_{in}, W_{in})` ，AvgPool3D在 :math:`(D_{in}, H_{in}, W_{in})` 维度上输出区域平均值。给定 `kernel_size` 为 :math:`ks = (d_{ker}, h_{ker}, w_{ker})` 和 `stride` :math:`s = (s_0, s_1, s_2)` ，运算如下：

    .. warning::
        "kernel_size"在[1, 255]范围中。"strides"在[1, 63]范围中。

    .. math::
        \text{output}(N_i, C_j, d, h, w) =
        \frac{1}{d_{ker} * h_{ker} * w_{ker}} \sum_{l=0}^{d_{ker}-1} \sum_{m=0}^{h_{ker}-1} \sum_{n=0}^{w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times d + l, s_1 \times h + m, s_2 \times w + n)

    参数：
        - **kernel_size** (Union[int, tuple[int]]) - 指定池化核尺寸大小，是一个整数，对应深度、高度和宽度，或者是含3个分别对应深度、高度和宽度整数的tuple。默认值： ``1`` 。
        - **strides** (Union[int, tuple[int]]) - 池化操作的移动步长，是一个整数，对应移动深度、高度和宽度，或者是含3个分别表对应移动深度、高度和宽度整数的tuple。默认值： ``1`` 。
        - **pad_mode** (str，可选) - 指定填充模式，填充值为0。可选值为 ``"same"`` ， ``"valid"`` 或 ``"pad"`` 。默认值： ``"valid"`` 。

          - ``"same"``：在输入的深度、高度和宽度维度进行填充，使得当 `stride` 为 ``1`` 时，输入和输出的shape一致。待填充的量由算子内部计算，若为偶数，则均匀地填充在四周，若为奇数，多余的填充量将补充在前方/底部/右侧。如果设置了此模式， `pad` 必须为0。
          - ``"valid"``：不对输入进行填充，返回输出可能的最大深度、高度和宽度，不能构成一个完整stride的额外的像素将被丢弃。如果设置了此模式， `pad` 必须为0。
          - ``"pad"``：对输入填充指定的量。在这种模式下，在输入的深度、高度和宽度方向上填充的量由 `pad` 参数指定。如果设置此模式， `pad` 必须大于或等于0。

        - **pad** (Union(int, tuple[int], list[int])) - 池化填充方式。默认值： ``0`` 。如果 `pad` 是一个整数，则头部、尾部、顶部、底部、左边和右边的填充都是相同的，等于 `pad` 。如果 `pad` 是六个integer的tuple，则头部、尾部、顶部、底部、左边和右边的填充分别等于填充pad[0]、pad[1]、pad[2]、pad[3]、pad[4]和pad[5]。
        - **ceil_mode** (bool) - 是否使用ceil函数计算输出高度和宽度。默认值： ``False`` 。
        - **count_include_pad** (bool) - 如果为 ``True`` ，平均计算将包括零填充。默认值： ``True`` 。
        - **divisor_override** (int) - 如果指定了该值，它将在平均计算中用作除数，否则将使用kernel_size作为除数。默认值： ``0`` 。
        - **data_format** (str) - 输入和输出的数据格式。目前仅支持'NCDHW'。默认值： ``"NCDHW"`` 。

    输入：
        - **x** (Tensor) - shape为 :math:`(N, C, D_{in}, H_{in}, W_{in})` 的Tensor。数据类型为float16、float32和float64。

    输出：
        Tensor，其shape为 :math:`(N, C, D_{out}, H_{out}, W_{out})` ，数据类型与 `x` 相同。

    异常：
        - **TypeError** - `kernel_size` 、 `strides` 或 `pad` 既不是int也不是tuple。
        - **TypeError** - `ceil_mode` 或 `count_include_pad` 不是bool。
        - **TypeError** - `pad_mode` 或 `data_format` 不是string。
        - **TypeError** - `divisor_override` 不是int。
        - **ValueError** - `kernel_size` 或 `strides` 中的数字不是正数。
        - **ValueError** - `kernel_size` 或 `strides` 是长度不等于3的tuple。
        - **ValueError** - `pad_mode` 不是'same'，'valid'，或'pad'。
        - **ValueError** - `pad` 是长度不等于6的tuple。
        - **ValueError** - `pad` 的元素小于0。
        - **ValueError** - `pad_mode` 不等于'pad'且 `pad` 不等于0或(0, 0, 0, 0, 0, 0)。
        - **ValueError** - `data_format` 不是'NCDHW'。