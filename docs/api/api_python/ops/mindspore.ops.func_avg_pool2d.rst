mindspore.ops.avg_pool2d
========================

.. py:function:: mindspore.ops.avg_pool2d(input_x, kernel_size=1, stride=1, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=0)

    对输入的多维数据进行二维平均池化运算。

    在输入Tensor上应用2D average pooling，可被视为2D输入平面。

    一般地，输入的shape为 :math:`(N_{in}, C_{in}, H_{in}, W_{in})` ，输出 :math:`(H_{in}, W_{in})` 维度的区域平均值。给定 `kernel_size` 为 :math:`(k_{h}, k_{w})` 和 `stride` ，运算如下：

    .. math::
        \text{output}(N_i, C_j, h, w) = \frac{1}{k_{h} * k_{w}} \sum_{m=0}^{k_{h}-1} \sum_{n=0}^{k_{w}-1}
        \text{input}(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)

    .. warning::
        - `kernel_size` 取值为[1, 255]范围内的正整数，`stride` 的取值为[1, 63]范围内的正整数。

    参数：
        - **input_x** (Tensor) - 输入shape为 :math:`(N, C_{in}, H_{in}, W_{in})` 的Tensor。
        - **kernel_size** (Union[int, tuple[int]]) - 指定池化核尺寸大小，可以是单个整数表示池化核高度和宽度，或者整数tuple分别表示池化核高度和宽度。默认值：1。
        - **stride** (Union[int, tuple[int]]) - 池化操作的移动步长，可以是单个整数表示高度和宽度方向的移动步长，或者整数tuple分别表示高度和宽度方向的移动步长。默认值：1。
        - **padding** (Union(int, tuple[int])) - 池化填充长度。可以是一个整数表示在上下左右方向的填充长度，或者包含四个整数的tuple，分别表示在上下左右方向的填充长度。默认值：0。
        - **ceil_mode** (bool) - 如果为True，用ceil代替floor来计算输出的shape。默认值：False。
        - **count_include_pad** (bool) - 如果为True，平均计算将包括零填充。默认值：True。
        - **divisor_override** (int) - 如果指定了该值，它将在平均计算中用作除数，否则将使用 `kernel_size` 作为除数。默认值：0。

    返回：
        Tensor，shape为 :math:`(N, C_{out}, H_{out}, W_{out})` 。

    异常：
        - **TypeError** - `input_x` 不是一个Tensor。
        - **TypeError** - `kernel_size` 或 `stride` 既不是int也不是tuple。
        - **TypeError** - `ceil_mode` 或 `count_include_pad` 不是bool。
        - **TypeError** - `divisor_override` 不是int。
        - **ValueError** - `input_x` 的shape长度不等于4。
        - **ValueError** - `kernel_size` 或 `stride` 小于1。
        - **ValueError** - `kernel_size` 或 `stride` 是长度不等于2的tuple。
        - **ValueError** - `padding` 不是int或者tuple的长度不等于4。
        - **ValueError** - `padding` 的值小于0。
