mindspore.nn.MaxPool3d
=======================

.. py:class:: mindspore.nn.MaxPool3d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

    对输入的多维数据进行三维的最大池化运算。

    在一个输入Tensor上应用3D max pooling，可被视为组成一个3D平面。

    通常，输入的shape为 :math:`(N_{in}, C_{in}, D_{in}, H_{in}, W_{in})` ，MaxPool3d输出 :math:`(D_{in}, H_{in}, W_{in})` 维度区域最大值。给定 `kernel_size` 为 :math:`ks = (d_{ker}, h_{ker}, w_{ker})` 和 `stride` 为 :math:`s = (s_0, s_1, s_2)`，公式如下。

    .. math::
        \text{output}(N_i, C_j, d, h, w) =
        \max_{l=0, \ldots, d_{ker}-1} \max_{m=0, \ldots, h_{ker}-1} \max_{n=0, \ldots, w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times d + l, s_1 \times h + m, s_2 \times w + n)

    参数：
        - **kernel_size** (Union[int, tuple[int]]) - 指定池化核尺寸大小，如果为整数，则代表池化核的深度，高和宽。如果为tuple，其值必须包含三个整数值分别表示池化核的深度，高和宽。默认值：1。
        - **stride** (Union[int, tuple[int]]) - 池化操作的移动步长，如果为整数，则代表池化核的深度，高和宽方向的移动步长。如果为tuple，其值必须包含三个整数值分别表示池化核的深度，高和宽的移动步长。默认值：1。
        - **padding** (Union[int, tuple[int]]) - 池化填充长度。可以是一个整数表示在深度，高度和宽度方向的填充长度，或者包含三个整数的tuple，分别表示在深度，高度和宽度方向的填充长度。
        - **dilation** (Union[int, tuple[int]]) - 控制池化核内元素的间距。默认值：1。
        - **return_indices** (bool) - 若为True，则返回一个包含两个Tensor的Tuple，表示池化的计算结果以及生成max值的位置，否则，仅返回池化计算结果。
        - **ceil_mode** (bool) - 若为True，使用ceil来计算输出shape。若为False，使用floor来计算输出shape。默认值：False。

    输入：
        - **x** (Tensor) - shape为 :math:`(N_{in}, C_{in}, D_{in}, H_{in}, W_{in})` 的Tensor。数据类型必须为int8、 int16、 int32、 int64、 uint8、 uint16、 uint32、 uint64、 float16、 float32 或者 float64。

    输出：
        如果 `return_indices` 为False，则是shape为 :math:`(N, C, D_{out}, H_{out}, W_{out})` 的Tensor。数据类型与 `x` 一致。
        如果 `return_indices` 为True，则是一个包含了两个Tensor的Tuple，表示maxpool的计算结果以及生成max值的位置。

        - **output** (Tensor) - 最大池化结果，shape为 :math:`(N_{out}, C_{out}, D_{out}, H_{out}, W_{out})`的Tensor。数据类型与 `x` 一致。
        - **argmax** (Tensor) - 最大值对应的索引。数据类型为int64。

    异常：
        - **TypeError** - `x` 不是一个Tensor。
        - **ValueError** - `x` 的shape长度不等于5。
        - **TypeError** - `kernel_size` 、 `stride` 、 `padding` 、 `dilation` 既不是整数也不是元组。
        - **ValueError** - `kernel_size` 或者 `stride` 小于1。
        - **ValueError** - `padding` 小于0。
