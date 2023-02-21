mindspore.nn.MaxPool1d
=======================

.. py:class:: mindspore.nn.MaxPool1d(kernel_size=1, stride=1, pad_mode="valid", padding=0, dilation=1, return_indices=False, ceil_mode=False)

    在一个输入Tensor上应用1D最大池化运算，该Tensor可被视为一维平面的组合。

    通常，输入的shape为 :math:`(N_{in}, C_{in}, L_{in})` ，MaxPool1d输出 :math:`(L_{in})` 维度区域最大值。
    给定 `kernel_size` 和 `stride` ，公式如下：

    .. math::
        \text{output}(N_i, C_j, l) = \max_{n=0, \ldots, kernel\_size-1}
        \text{input}(N_i, C_j, stride \times l + n)


    参数：
        - **kernel_size** (int) - 指定池化核尺寸大小。默认值：1。
        - **stride** (int) - 池化操作的移动步长，数据类型为整型。默认值：1。
        - **pad_mode** (str) - 指定池化填充模式，取值为"same"、"valid"或者"pad"，不区分大小写。默认值："valid"。

          - **same** - 输出的宽度与输入整数 `stride` 后的值相同。
          - **valid** - 在不填充的前提下返回有效计算所得的输出。不满足计算的多余像素会被丢弃。
          - **pad** - 对输入进行填充。在输入的左右两端填充 `padding` 大小的0。如果设置此模式， `padding` 必须大于或等于0。

        - **padding** (Union(int, tuple[int], list[int])) - 池化填充值。默认值：0。 `padding` 只能是一个整数或者包含一个整数的tuple/list，设定后，则会在输入的左边和右边填充 `padding` 次或者 `padding[0]` 次。
        - **dilation** (Union(int, tuple[int])) - 卷积核中各个元素之间的间隔大小，用于提升池化操作的感受野。默认值：1。
        - **return_indices** (bool) - 若为True，将会同时返回最大池化的结果和索引。默认值：False。
        - **ceil_mode** (bool) - 若为True，使用ceil来计算输出shape。若为False，使用floor来计算输出shape。默认值：False。

    输入：
        - **x** (Tensor) - shape为 :math:`(N, C_{in}, L_{in})` 或 :math:`(C_{in}, L_{in})` 的Tensor。

    输出：
        如果 `return_indices` 为False，则是shape为 :math:`(N, C_{out}, L_{out})` 或 :math:`(C_{out}, L_{out})` 的Tensor。数据类型与 `x` 一致。
        如果 `return_indices` 为True，则是一个包含了两个Tensor的Tuple，表示maxpool的计算结果以及生成max值的位置。

        - **output** (Tensor) - 最大池化结果，shape为 :math:`(N, C_{out}, L_{out})` 或 :math:`(C_{out}, L_{out})` 的Tensor。数据类型与 `x` 一致。
        - **argmax** (Tensor) - 最大值对应的索引。数据类型为int64。

    异常：
        - **TypeError** - `kernel_size` 或 `strides` 不是整数。
        - **ValueError** - `pad_mode` 既不是'valid'，也不是'same' 或者 'pad'，不区分大小写。
        - **ValueError** - `data_format` 既不是'NCHW'也不是'NHWC'。
        - **ValueError** - `kernel_size` 或 `strides` 小于1。
        - **ValueError** - `x` 的shape长度不等于2或3。
        - **ValueError** - 当 `pad_mode` 不为 'pad' 时，`padding`、 `dilation`、 `return_indices`、 `ceil_mode` 参数不为默认值。
        - **ValueError** - `padding` 参数为tuple/list时长度不为1。
        - **ValueError** - `dilation` 参数为tuple时长度不为1。
        - **ValueError** - `dilation` 参数不为int也不为tuple。