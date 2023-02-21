mindspore.nn.MaxPool3d
=======================

.. py:class:: mindspore.nn.MaxPool3d(kernel_size=1, stride=1, pad_mode="valid", padding=0, dilation=1, return_indices=False, ceil_mode=False)

    在一个输入Tensor上应用3D最大池化运算，输入Tensor可看成是由一系列3D平面组成的。

    通常，输入的shape为 :math:`(N_{in}, C_{in}, D_{in}, H_{in}, W_{in})` ，MaxPool3d输出 :math:`(D_{in}, H_{in}, W_{in})` 维度区域最大值。给定 `kernel_size` 为 :math:`ks = (d_{ker}, h_{ker}, w_{ker})` 和 `stride` 为 :math:`s = (s_0, s_1, s_2)`，公式如下。

    .. math::
        \text{output}(N_i, C_j, d, h, w) =
        \max_{l=0, \ldots, d_{ker}-1} \max_{m=0, \ldots, h_{ker}-1} \max_{n=0, \ldots, w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times d + l, s_1 \times h + m, s_2 \times w + n)

    参数：
        - **kernel_size** (Union[int, tuple[int]]) - 指定池化核尺寸大小，如果为int，则代表池化核的深度，高和宽。如果为tuple，其值必须包含三个正整数值分别表示池化核的深度，高和宽。取值必须为正整数。默认值：1。
        - **stride** (Union[int, tuple[int]]) - 池化操作的移动步长，如果为int，则代表池化核的深度，高和宽方向的移动步长。如果为tuple，其值必须包含三个正整数值分别表示池化核的深度，高和宽的移动步长。取值必须为正整数。如果值为None，则使用默认值 `kernel_size`。默认值：1。
        - **pad_mode** (str) - 指定池化填充模式，取值为"same"、"valid"或者"pad"，不区分大小写。默认值："valid"。

          - **same** - 输出的shape与输入的shape整除 `stride` 后的值相同。
          - **valid** - 在不填充的前提下返回有效计算所得的输出。不满足计算的多余像素会被丢弃。
          - **pad** - 对输入进行填充。在输入的前后上下左右分别填充 `padding` 大小的0。如果设置此模式， `padding` 必须大于或等于0。

        - **padding** (Union(int, tuple[int], list[int])) - 池化填充值。默认值：0。 `padding` 只能是一个整数或者包含一个或三个整数的tuple/list，若 `padding` 为一个整数或包含一个整数的tuple/list，则会分别在输入的前后上下左右六个方向进行 `padding` 次的填充，若 `padding` 为一个包含三个整数的tuple/list，则会在输入的前后进行 `padding[0]` 次的填充，上下进行 `padding[1]` 次的填充，在输入的左右进行 `padding[2]` 次的填充。
        - **dilation** (Union(int, tuple[int])) - 卷积核中各个元素之间的间隔大小，用于提升池化操作的感受野。如果为tuple，其值必须包含三个整数。默认值：1。
        - **return_indices** (bool) - 若为True，则返回一个包含两个Tensor的Tuple，表示池化的计算结果以及生成max值的位置，否则，仅返回池化计算结果。默认值：False。
        - **ceil_mode** (bool) - 若为True，使用ceil模式来计算输出shape。若为False，使用floor模式来计算输出shape。默认值：False。

    输入：
        - **x** (Tensor) - shape为 :math:`(N_{in}, C_{in}, D_{in}, H_{in}, W_{in})` 或者 :math:`(C_{in}, D_{in}, H_{in}, W_{in})` 的Tensor。

    输出：
        如果 `return_indices` 为False，则是shape为 :math:`(N, C, D_{out}, H_{out}, W_{out})` 或者 :math:`(C_{in}, D_{in}, H_{in}, W_{in})` 的Tensor。数据类型与 `x` 一致。
        如果 `return_indices` 为True，则是一个包含了两个Tensor的Tuple，表示maxpool的计算结果以及生成max值的位置。

        - **output** (Tensor) - 最大池化结果，shape为 :math:`(N_{out}, C_{out}, D_{out}, H_{out}, W_{out})` 或者 :math:`(C_{in}, D_{in}, H_{in}, W_{in})` 的Tensor。数据类型与 `x` 一致。
        - **argmax** (Tensor) - 最大值对应的索引。数据类型为int64。

    异常：
        - **ValueError** - `x` 的shape长度不等于 4 或 5。
        - **TypeError** - `kernel_size` 、 `stride` 、 `padding` 、 `dilation` 既不是整数也不是元组。
        - **ValueError** - `kernel_size` 或者 `stride` 小于1。
        - **ValueError** - `padding` 不为int也不为长度为3的tuple。
        - **ValueError** - `pad_mode` 不为 'pad' 模式时，`return_indices` 设为了True或者 `dilation` 不为1。
