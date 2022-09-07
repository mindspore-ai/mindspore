mindspore.nn.AvgPool1d
=======================

.. py:class:: mindspore.nn.AvgPool1d(kernel_size=1, stride=1, pad_mode='valid')

    对输入的多维数据进行一维平面上的平均池化运算。

    在一个输入Tensor上应用1D average pooling，可被视为组成一个1D输入平面。

    通常，输入的shape为 :math:`(N_{in}, C_{in}, L_{in})` ，AvgPool1d在 :math:`(L_{in})` 维度上输出区域平均值。
    给定 `kernel_size` 为 :math:`k` 和 `stride` ，公式定义如下：

    .. math::
        \text{output}(N_i, C_j, l) = \frac{1}{k} \sum_{n=0}^{k-1}
        \text{input}(N_i, C_j, stride \times l + n)

    .. note::
        pad_mode仅支持"same"和"valid"。

    参数：
        - **kernel_size** (int) - 指定池化核尺寸大小，数据类型为整型。默认值：1。
        - **stride** (int) - 池化操作的移动步长，数据类型为整型。默认值：1。
        - **pad_mode** (str) - 指定池化的填充方式，可选值为"same"或"valid"，不区分大小写。默认值："valid"。

          - **same** - 输出的shape与输入整数 `stride` 后的值相同。
          - **valid** - 在不填充的前提下返回有效计算所得的输出。不满足计算的多余像素会被丢弃。

    输入：
        - **x** (Tensor) - shape为 :math:`(N, C_{in}, L_{in})` 的Tensor。

    输出：
        shape为 :math:`(N, C_{out}, L_{out})` 的Tensor。

    异常：
        - **TypeError** - `kernel_size` 或 `stride` 不是int。
        - **ValueError** - `pad_mode` 既不是"valid"，也不是"same"，不区分大小写。
        - **ValueError** - `kernel_size` 或 `stride` 小于1。
        - **ValueError** -  `x` 的shape长度不等于3。
