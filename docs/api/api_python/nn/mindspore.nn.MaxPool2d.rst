mindspore.nn.MaxPool2d
=======================

.. py:class:: mindspore.nn.MaxPool2d(kernel_size=1, stride=1, pad_mode='valid', data_format='NCHW')

    对输入的多维数据进行二维的最大池化运算。

    在一个输入Tensor上应用2D max pooling，可被视为组成一个2D平面。

    通常，输入的形状为 :math:`(N_{in}, C_{in}, H_{in}, W_{in})` ，MaxPool2d输出 :math:`(H_{in}, W_{in})` 维度区域最大值。给定 `kernel_size` 为 :math:`(kH,kW)` 和 `stride` ，公式如下。

    .. math::
        \text{output}(N_i, C_j, h, w) = \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1}
        \text{input}(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)

    .. note::
        pad_mode仅支持"same"和"valid"。

    参数：
        - **kernel_size** (Union[int, tuple[int]]) - 指定池化核尺寸大小，如果为整数，则代表池化核的高和宽。如果为tuple，其值必须包含两个整数值分别表示池化核的高和宽。默认值：1。
        - **stride** (Union[int, tuple[int]]) - 池化操作的移动步长，如果为整数，则代表池化核的高和宽方向的移动步长。如果为tuple，其值必须包含两个整数值分别表示池化核的高和宽的移动步长。默认值：1。
        - **pad_mode** (str) - 指定池化填充模式，取值为"same"或"valid"，不区分大小写。默认值："valid"。

          - **same** - 输出的shape与输入的shape整除 `stride` 后的值相同。
          - **valid** - 在不填充的前提下返回有效计算所得的输出。不满足计算的多余像素会被丢弃。

        - **data_format** (str) - 输入数据格式可为'NHWC'或'NCHW'。默认值：'NCHW'。

    输入：
        - **x** (Tensor) - shape为 :math:`(N,C_{in},H_{in},W_{in})` 的Tensor。

    输出：
        shape为 :math:`(N,C_{out},H_{out},W_{out})` 的Tensor。

    异常：
        - **TypeError** - `kernel_size` 或 `strides` 既不是整数也不是元组。
        - **ValueError** - `pad_mode` 既不是'valid'，也不是'same'，不区分大小写。
        - **ValueError** - `data_format` 既不是'NCHW'也不是'NHWC'。
        - **ValueError** - `kernel_size` 或 `strides` 小于1。
        - **ValueError** - `x` 的shape长度不等于4。
