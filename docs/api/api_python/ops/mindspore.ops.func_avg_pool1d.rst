mindspore.ops.avg_pool1d
========================

.. py:function:: mindspore.ops.avg_pool1d(input_x, kernel_size=1, stride=1, padding=0, ceil_mode=False, count_include_pad=True)

    在输入Tensor上应用1D平均池化，输入Tensor可以看作是由一系列1D平面组成的。

    一般地，输入的shape为 :math:`(N_{in}, C_{in}, L_{in})` ，输出 :math:`(L_{in})` 维度的区域平均值。给定 `kernel_size` 为 :math:`ks = l_{ker}` 和 `stride` 为 :math:`s = s_0` ，运算如下：

    .. math::
        \text{output}(N_i, C_j, l) = \frac{1}{l_{ker}} \sum_{n=0}^{l_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times l + n)

    .. warning::
        - `kernel_size` 取值为[1, 255]范围内的正整数，`stride` 的取值为[1, 63]范围内的正整数。

    参数：
        - **input_x** (Tensor) - 输入shape为 :math:`(N, C_{in}, L_{in})` 的Tensor。
        - **kernel_size** (int) - 指定池化核尺寸大小。默认值：1。
        - **stride** (Union(int, tuple[int])) - 池化操作的移动步长，可以是单个整数表示高度和宽度方向的移动步长，或者整数tuple分别表示高度和宽度方向的移动步长。默认值：1。
        - **padding** (Union(int, tuple[int])) - 池化填充长度。可以是一个整数表示在左右方向的填充长度，或者包含两个整数的tuple，分别表示在左右方向的填充长度。默认值：0。
        - **ceil_mode** (bool) - 如果为True，用ceil代替floor来计算输出的shape。默认值：False。
        - **count_include_pad** (bool) - 如果为True，平均计算将包括零填充。默认值：True。

    返回：
        Tensor，shape为 :math:`(N, C_{out}, L_{out})` 。

    异常：
        - **TypeError** - `input_x` 不是一个Tensor。
        - **TypeError** - `kernel_size` 或 `stride` 不是int。
        - **TypeError** - `ceil_mode` 或 `count_include_pad` 不是bool。
        - **ValueError** - `input_x` 的shape长度不等于3。
        - **ValueError** - `kernel_size` 或 `stride` 小于1。
        - **ValueError** - `padding` 不是int或者tuple的长度不等于2。
        - **ValueError** - `padding` 的值小于0。
