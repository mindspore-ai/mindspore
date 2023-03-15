mindspore.nn.MaxPool2d
=======================

.. py:class:: mindspore.nn.MaxPool2d(kernel_size=1, stride=1, pad_mode="valid", padding=0, dilation=1, return_indices=False, ceil_mode=False, data_format="NCHW")

    在一个输入Tensor上应用2D最大池化运算，可被视为组成一个2D平面。

    通常，输入的shape为 :math:`(N_{in}, C_{in}, H_{in}, W_{in})` ，MaxPool2d输出 :math:`(H_{in}, W_{in})` 维度区域最大值。给定 `kernel_size` 为 :math:`(h_{ker}, w_{ker})` ， `stride` 为 :math:`(s_0, s_1)`，公式如下。

    .. math::
        \text{output}(N_i, C_j, h, w) = \max_{m=0, \ldots, h_{ker}-1} \max_{n=0, \ldots, w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times h + m, s_1 \times w + n)

    参数：
        - **kernel_size** (Union[int, tuple[int]]) - 指定池化核尺寸大小，如果为整数，则代表池化核的高和宽。如果为tuple，其值必须包含两个整数值分别表示池化核的高和宽。默认值：1。
        - **stride** (Union[int, tuple[int]]) - 池化操作的移动步长，如果为整数，则代表池化核的高和宽方向的移动步长。如果为tuple，其值必须包含两个整数值分别表示池化核的高和宽的移动步长。默认值：1。
        - **pad_mode** (str) - 指定池化填充模式，取值为"same"、"valid"或者"pad"，不区分大小写。默认值："valid"。

          - **same** - 输出的shape与输入的shape整除 `stride` 后的值相同。
          - **valid** - 在不填充的前提下返回有效计算所得的输出。不满足计算的多余像素会被丢弃。
          - **pad** - 对输入进行填充。在输入的上下左右分别填充 `padding` 大小的0。如果设置此模式， `padding` 必须大于或等于0。

        - **padding** (Union(int, tuple[int], list[int])) - 池化填充值。默认值：0。 `padding` 只能是一个整数或者包含一个或两个整数的元组，若 `padding` 为一个整数或者包含一个整数的tuple/list，则会分别在输入的上下左右四个方向进行 `padding` 次的填充，若 `padding` 为一个包含两个整数的tuple/list，则会在输入的上下进行 `padding[0]` 次的填充，在输入的左右进行 `padding[1]` 次的填充。
        - **dilation** (Union(int, tuple[int])) - 卷积核中各个元素之间的间隔大小，用于提升池化操作的感受野。如果为tuple，其值必须包含两个整数。默认值：1。
        - **return_indices** (bool) - 若为True，将会同时返回最大池化的结果和索引。默认值：False。
        - **ceil_mode** (bool) - 若为True，使用ceil来计算输出shape。若为False，使用floor来计算输出shape。默认值：False。
        - **data_format** (str) - 输入数据格式可为'NHWC'或'NCHW'。默认值：'NCHW'。

    输入：
        - **x** (Tensor) - 输入数据的shape为 :math:`(N,C_{in},H_{in},W_{in})` 或 :math:`(C_{in},H_{in},W_{in})` 的Tensor。

    输出：
        如果 `return_indices` 为False，则是shape为 :math:`(N, C, H_{out}, W_{out})` 或者 :math:`(C_{out}, H_{out}, W_{out})` 的Tensor。数据类型与 `x` 一致。
        如果 `return_indices` 为True，则是一个包含了两个Tensor的Tuple，表示maxpool的计算结果以及生成max值的位置。

        - **output** (Tensor) - 最大池化结果，shape为 :math:`(N_{out}, C_{out}, H_{out}, W_{out})` 或者 :math:`(C_{out}, H_{out}, W_{out})` 的Tensor。数据类型与 `x` 一致。
        - **argmax** (Tensor) - 最大值对应的索引。数据类型为int64。

    异常：
        - **TypeError** - `kernel_size` 或 `strides` 既不是整数也不是元组。
        - **ValueError** - `pad_mode` 既不是'valid'，也不是'same' 或者 'pad'，不区分大小写。
        - **ValueError** - `data_format` 既不是'NCHW'也不是'NHWC'。
        - **ValueError** - `kernel_size` 或 `strides` 小于1。
        - **ValueError** - `x` 的shape长度不等于3或4。
        - **ValueError** - 当 `pad_mode` 不为 'pad' 时，`padding`、 `dilation`、 `return_indices`、 `ceil_mode` 参数不为默认值。
        - **ValueError** - `padding` 参数为tuple/list时长度不为2。
        - **ValueError** - `dilation` 参数为tuple时长度不为2。
        - **ValueError** - `dilation` 参数不为int也不为tuple。
        - **ValueError** - `pad_mode` 为 'pad' 时，`data_format` 为 'NHWC'。