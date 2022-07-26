mindspore.ops.MaxPool3D
========================

.. py:class:: mindspore.ops.MaxPool3D(kernel_size=1, strides=1, pad_mode="VALID", pad_list=0, ceil_mode=None, data_format="NCDHW")

    对输入的多维数据进行三维的最大池化运算。

    一般，输入shape为 :math:`(N_{in}, C_{in}, D_{in}, H_{in}, W_{in})` 的Tensor，输出 :math:`(D_{in}, H_{in}, W_{in})` 维上的区域最大值。给定 `kernel_size` 为 :math:`(kD,kH,kW)` 和 `stride` ，运算如下：

    .. math::
        \text{output}(N_i, C_j, d, h, w) =
        \max_{l=0, \ldots, kD-1} \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1}
        \text{input}(N_i, C_j, stride[0] \times d + l, stride[1] \times h + m, stride[2] \times w + n)

    参数：
        - **kernel_size** (Union[int, tuple[int]]) - 指定池化核尺寸大小。整数类型，表示池化核深度、高和宽，或者是三个整数组成的元组，表示深、高和宽。默认值：1。
        - **strides** (Union[int, tuple[int]]) - 池化操作的移动步长，整数类型，表示深、高和宽的移动步长，或者是三个整数组成的元组，表示深、高和宽移动步长。默认值：1。
        - **pad_mode** (str) - 指定池化填充模式，可选值有："same"、"valid"或"pad"。默认值："valid"。

          - same：输出的宽度于输入整数 `stride` 后的值相同。
          - valid：在不填充的前提下返回有效计算所得的输出。不满足计算的多余像素会被丢弃。
          - pad：对输入进行填充。在输入的深度、高度和宽度方向上填充 `pad` 大小的0。如果设置此模式， `pad_list` 必须大于或等于0。

        - **pad_list** (Union(int, tuple[int])) - 池化填充方式。默认值：0。如果 `pad` 是一个整数，则头尾部、顶部，底部，左边和右边的填充都是相同的，等于 `pad` 。如果 `pad` 是六个整数的tuple，则头尾部、顶部、底部、左边和右边的填充分别等于填充pad[0]、pad[1]、pad[2]、pad[3]、pad[4]和pad[5]。  
        - **ceil_mode** (Union[bool, None]) - 是否使用ceil函数计算输出高度和宽度。默认值：None。
        - **data_format** (str) - 输入和输出的数据格式。目前仅支持'NCDHW'。默认值：'NCDHW'。

    输入：
        - **x** (Tensor) - shape为 :math:`(N, C, D_{in}, H_{in}, W_{in})` 的Tensor。数据类型为float16或float32。

    输出：
        Tensor，shape为 :math:`(N, C, D_{out}, H_{out}, W_{out})` 。数据类型与 `x` 相同。

    异常：
        - **TypeError** - `kernel_size` 或 `strides` 既不是int也不是元组。
        - **TypeError** - `pad_mode` 或 `data_format` 不是str。
        - **ValueError** - `kernel_size` 或 `strides` 不是正数。
        - **ValueError** - `pad_mode` 不是'same'，'valid'，或'pad'。
        - **ValueError** - `pad_mode` 取值为'same'或'valid'， `ceil_mode` 取值不是None。
        - **ValueError** - `kernel_size` 或 `strides` 是长度不等于3的元组。
        - **ValueError** - `data_format` 不是'NCDHW'。