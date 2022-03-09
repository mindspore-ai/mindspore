mindspore.ops.AvgPool
======================

.. py:class:: mindspore.ops.AvgPool(kernel_size=1, strides=1, pad_mode="valid", data_format="NCHW")

    对输入的多维数据进行二维平均池化运算。

    一般地，输入的形状为 :math:`(N_{in}, C_{in}, H_{in}, W_{in})` ，AvgPool输出 :math:`(H_{in}, W_{in})` 维度的区域平均值。给定 `kernel_size` 为 :math:`(kH, kW)` 和 `stride` ，运算如下：

    .. math::
        \text{output}(N_i, C_j, h, w) = \frac{1}{kH * kW} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1}
        \text{input}(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)

    .. warning::
        - 支持全局池化。
        - 在Ascend上，"kernel_size"的高度和权重取值为[1, 255]范围内的正整数。 :math: ksize_H * ksize_W < 256。
        - 由于指令限制，"strides_h"和"strides_w"的取值为[1, 63]范围内的正整数。

    **参数：**

    - **kernel_size** (Union[int, tuple[int]]) - 指定池化核尺寸大小，表示内核高度和宽度的整数值，或者是两个分别表示高度和宽度的整数tuple。默认值：1。
    - **strides** (Union[int, tuple[int]]) - 池化操作的移动步长，表示移动高度和宽度的整数都是步长，或者两个分别表示移动高度和宽度的整数tuple。默认值：1。
    - **pad_mode** (str) - 指定池化填充模式，取值为"same"或"valid"，不区分大小写。默认值："valid"。

      - **same** - 输出的高度和宽度分别与输入整除 `stride` 后的值相同。
      - **valid** - 在不填充的前提下返回有效计算所得的输出。不满足计算的多余像素会被丢弃。

    - **data_format** (str) - 指定输入和输出的数据格式。取值为'NHWC'或'NCHW'。默认值：'NCHW'。

    **输入：**

    - **x** (Tensor) - 输入shape为 :math:`(N, C_{in}, H_{in}, W_{in})` 的Tensor。

    **输出：**

    Tensor，shape为 :math:`(N,C_{out},H_{out},W_{out})` 。

    **异常：**

    - **TypeError** - `kernel_size` 或 `strides` 既不是int也不是tuple。
    - **ValueError** - `pad_mode` 既不是'valid'，也不是'same'，不区分大小写。
    - **ValueError** - `data_format` 既不是'NCHW'也不是'NHWC'。
    - **ValueError** - `kernel_size` 或 `strides` 小于1。
    - **ValueError** - `x` 的shape长度不等于4。