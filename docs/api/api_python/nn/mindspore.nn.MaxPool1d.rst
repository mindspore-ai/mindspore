mindspore.nn.MaxPool1d
=======================

.. py:class:: mindspore.nn.MaxPool1d(kernel_size=1, stride=1, pad_mode='valid')

    在一个输入Tensor上应用1D最大池化运算，该Tensor可被视为一维平面的组合。

    通常，输入的shape为 :math:`(N_{in}, C_{in}, L_{in})` ，MaxPool1d输出 :math:`(L_{in})` 维度区域最大值。
    给定 `kernel_size` 和 `stride` ，公式如下：

    .. math::
        \text{output}(N_i, C_j, l) = \max_{n=0, \ldots, kernel\_size-1}
        \text{input}(N_i, C_j, stride \times l + n)

    .. note::
         pad_mode仅支持"same"和"valid"。

    参数：
        - **kernel_size** (int) - 指定池化核尺寸大小。默认值：1。
        - **stride** (int) - 池化操作的移动步长，数据类型为整型。默认值：1。
        - **pad_mode** (str) - 指定池化填充模式，取值为"same"或"valid"，不区分大小写。默认值："valid"。

          - **same** - 输出的宽度与输入整数 `stride` 后的值相同。
          - **valid** - 在不填充的前提下返回有效计算所得的输出。不满足计算的多余像素会被丢弃。

    输入：
        - **x** (Tensor) - shape为 :math:`(N, C, L_{in})` 的Tensor。

    输出：
        shape为 :math:`(N, C, L_{out})` 的Tensor。

    异常：
        - **TypeError** - `kernel_size` 或 `strides` 不是整数。
        - **ValueError** - `pad_mode` 既不是'valid'，也不是'same'，不区分大小写。
        - **ValueError** - `data_format` 既不是'NCHW'也不是'NHWC'。
        - **ValueError** - `kernel_size` 或 `strides` 小于1。
        - **ValueError** - `x` 的shape长度不等于4。
