mindspore.ops.avg_pool3d
========================

.. py:function:: mindspore.ops.avg_pool3d(input_x, kernel_size=1, stride=1, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=0)

    在输入Tensor上应用3D平均池化，输入Tensor可以看作是由一系列3D平面组成的。

    一般地，输入的shape为 :math:`(N, C, D_{in}, H_{in}, W_{in})` ，输出 :math:`(D_{in}, H_{in}, W_{in})` 维度的区域平均值。给定 `kernel_size` 为 :math:`ks = (d_{ker}, h_{ker}, w_{ker})` 和 `stride` 为 :math:`s = (s_0, s_1, s_2)`，运算如下：

    .. math::
        \text{output}(N_i, C_j, d, h, w) =
        \frac{1}{d_{ker} * h_{ker} * w_{ker}} \sum_{l=0}^{d_{ker}-1} \sum_{m=0}^{h_{ker}-1} \sum_{n=0}^{w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times d + l, s_1 \times h + m, s_2 \times w + n)

    .. warning::
        - `kernel_size` 取值为[1, 255]范围内的正整数，`stride` 的取值为[1, 63]范围内的正整数。

    参数：
        - **input_x** (Tensor) - 输入shape为 :math:`(N, C, D_{in}, H_{in}, W_{in})` 的Tensor，数据类型为float16和float32。
        - **kernel_size** (Union[int, tuple[int]], 可选) - 指定池化核尺寸大小，可以是单个整数表示池化核深度、高度、宽度，或者整数tuple分别表示池化核深度、高度、宽度。默认值：1。
        - **stride** (Union[int, tuple[int]], 可选) - 池化操作的移动步长，可以是单个整数表示深度、高度、宽度方向的移动步长，或者整数tuple分别表示深度、高度、深度方向的移动步长。默认值：1。
        - **padding** (Union(int, tuple[int]), 可选) - 池化填充长度。可以是一个整数表示在头尾上下左右方向的填充长度，或者包含六个整数的tuple，分别表示在头尾上下左右方向的填充长度。默认值：0。
        - **ceil_mode** (bool, 可选) - 如果为True，用ceil代替floor来计算输出的shape。默认值：False。
        - **count_include_pad** (bool, 可选) - 如果为True，平均计算将包括零填充。默认值：True。
        - **divisor_override** (int, 可选) - 如果指定了该值，它将在平均计算中用作除数，否则将使用 `kernel_size` 作为除数。默认值：0。

    返回：
        Tensor，shape为 :math:`(N, C, D_{out}, H_{out}, W_{out})` ，数据类型与 `input_x` 一致。

    异常：
        - **TypeError** - `input_x` 不是一个Tensor。
        - **TypeError** - `kernel_size` 或 `stride` 或 `padding` 既不是int也不是tuple。
        - **TypeError** - `ceil_mode` 或 `count_include_pad` 不是bool。
        - **TypeError** - `divisor_override` 不是int。
        - **ValueError** - `input_x` 的shape长度不等于5。
        - **ValueError** - `kernel_size` 或 `stride` 的值不是正整数。
        - **ValueError** - `kernel_size` 或 `stride` 是长度不等于3的tuple。
        - **ValueError** - `padding` 是tuple时，其长度不等于6。
        - **ValueError** - `padding` 的值小于0。
