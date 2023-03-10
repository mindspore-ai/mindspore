mindspore.ops.Conv2D
====================

.. py:class:: mindspore.ops.Conv2D(out_channel, kernel_size, mode=1, pad_mode="valid", pad=0, stride=1, dilation=1, group=1, data_format="NCHW")

    二维卷积层。

    对输入Tensor计算二维卷积，该Tensor的常见shape为 :math:`(N, C_{in}, H_{in}, W_{in})` ，其中 :math:`N` 为batch size，:math:`C_{in}` 为通道数， :math:`H_{in}, W_{in}` 分别为特征层的高度和宽度。 :math:`X_i` 为 :math:`i^{th}` 输入值， :math:`b_i` 为 :math:`i^{th}` 输入值的偏置项。对于每个batch中的Tensor，其shape为 :math:`(C_{in}, H_{in}, W_{in})` ，公式定义如下：

    .. math::
        out_j = \sum_{i=0}^{C_{in} - 1} ccor(W_{ij}, X_i) + b_j,

    其中， :math:`ccor` 为 `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_ ， :math:`C_{in}` 为输入通道数， :math:`j` 的范围从 :math:`0` 到 :math:`C_{out} - 1` ， :math:`W_{ij}` 对应第 :math:`j` 个过滤器的第 :math:`i` 个通道， :math:`out_{j}` 对应输出的第 :math:`j` 个通道。 :math:`W_{ij}` 为卷积核的切片，其shape为 :math:`(\text{kernel_size[0]},\text{kernel_size[1]})` ，其中 :math:`\text{kernel_size[0]}` 和 :math:`\text{kernel_size[1]}` 是卷积核的高度和宽度。完整卷积核的shape为 :math:`(C_{out}, C_{in} / \text{group}, \text{kernel_size[0]}, \text{kernel_size[1]})` ，其中 `group` 是在通道上分割输入 `x` 的组数。

    如果 `pad_mode` 设置为"pad"，则输出高度和宽度将分别为 :math:`\left \lfloor{1 + \frac{H_{in} + \text{padding[0]} + \text{padding[1]} - \text{kernel_size[0]} - (\text{kernel_size[0]} - 1) \times (\text{dilation[0]} - 1) }{\text{stride[0]}}} \right \rfloor` 和 :math:`\left \lfloor{1 + \frac{W_{in} + \text{padding[2]} + \text{padding[3]} - \text{kernel_size[1]} - (\text{kernel_size[1]} - 1) \times (\text{dilation[1]} - 1) }{\text{stride[1]}}} \right \rfloor` 。
    其中， :math:`dialtion` 为卷积核元素之间的间距， :math:`stride` 为移动步长， :math:`padding` 为添加到输入两侧的零填充。

    请参考论文 `Gradient Based Learning Applied to Document Recognition <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_ 。更详细的介绍，参见：http://cs231n.github.io/convolutional-networks/。

    参数：
        - **out_channel** (int) - :math:`C_{out}` 为输出通道数。
        - **kernel_size** (Union[int, tuple[int]]) - 数据类型为int或一个包含2个int组成的元组。指定二维卷积核的高度和宽度。单个整数表示该值同时适用于内核的高度和宽度。包含2个整数的元组表示第一个值用于高度，另一个值用于内核的宽度。
        - **mode** (int) - 指定不同的卷积模式。此值目前未被使用。默认值：1。
        - **pad_mode** (str) - 指定填充模式。取值为"same"，"valid"，或"pad"。默认值："valid"。

          - **same**: 输出的高度和宽度分别与输入整除 `stride` 后的值相同。填充将被均匀地添加到高和宽的两侧，剩余填充量将被添加到维度末端。若设置该模式，`pad` 的值必须为0。
          - **valid**: 在不填充的前提下返回有效计算所得的输出。不满足计算的多余像素会被丢弃。如果设置此模式，则 `pad` 的值必须为0。
          - **pad**: 对输入 `x` 进行填充。在输入的高度和宽度方向上填充 `pad` 大小的0。如果设置此模式， `pad` 必须大于或等于0。

        - **pad** (Union(int, tuple[int])) - 输入 `x` 的高度和宽度方向上填充的数量。数据类型为int或包含4个int组成的tuple。如果 `pad` 是一个int，那么上、下、左、右的填充都等于 `pad` 。如果 `pad` 是一个有4个int组成的tuple，那么上、下、左、右的填充分别等于 `pad[0]` 、 `pad[1]` 、 `pad[2]` 和 `pad[3]` 。值应该要大于等于0，默认值：0。
        - **stride** (Union(int, tuple[int])) - 卷积核移动的步长，数据类型为int或两个int组成的tuple。一个int表示在高度和宽度方向的移动步长均为该值。两个int组成的tuple分别表示在高度和宽度方向的移动步长。默认值：1。
        - **dilation** (Union(int, tuple[int])) - 卷积核膨胀尺寸。数据类型为int或由2个int组成的tuple。若 :math:`k > 1` ，则卷积核间隔 `k` 个元素进行采样。垂直和水平方向上的 `k` ，其取值范围分别为[1, H]和[1, W]。默认值：1。
        - **group** (int) - 将过滤器拆分为组。默认值：1。
        - **data_format** (str) - 数据格式的可选值有"NHWC"，"NCHW"。默认值："NCHW"。

    输入：
        - **x** (Tensor) - shape为 :math:`(N, C_{in}, H_{in}, W_{in})` 的Tensor。
        - **weight** (Tensor) - 设置卷积核的大小为 :math:`(\text{kernel_size[0]}, \text{kernel_size[1]})` ，则shape为 :math:`(C_{out}, C_{in}, \text{kernel_size[0]}, \text{kernel_size[1]})` 。

    输出：
        Tensor，卷积后的值。shape为 :math:`(N, C_{out}, H_{out}, W_{out})` 。

    异常：
        - **TypeError** - `kernel_size` 、 `stride` 、 `pad` 或 `dilation` 既不是int也不是tuple。
        - **TypeError** - `out_channel` 或 `group` 不是int。
        - **ValueError** - `kernel_size` 、 `stride` 或 `diation` 小于1。
        - **ValueError** - `pad_mode` 不是"same"、"valid"或"pad"。
        - **ValueError** - `pad` 是一个长度不等于4的tuple。
        - **ValueError** - `pad_mode` 不等于"pad"，`pad` 不等于(0, 0, 0, 0)。
        - **ValueError** - `data_format` 既不是"NCHW"，也不是"NHWC"。
