mindspore.ops.conv1d
====================

.. py:function:: mindspore.ops.conv1d(input, weight, bias=None, stride=1, pad_mode="valid", padding=0, dilation=1, groups=1)

    对输入Tensor计算一维卷积。该Tensor的常见shape为 :math:`(N, C_{in}, W_{in})` ，其中 :math:`N` 为batch size，:math:`C_{in}` 为通道数， :math:`W_{in}` 分别为特征层的宽度， :math:`X_i` 为 :math:`i^{th}` 输入值， :math:`b_i` 为 :math:`i^{th}` 输入值的偏置项。对于每个batch中的Tensor，其shape为 :math:`(C_{in}, W_{in})` ，公式定义如下：

    .. math::
        out_j = \sum_{i=0}^{C_{in} - 1} ccor(W_{ij}, X_i) + b_j,

    其中， :math:`ccor` 为 `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_ ， :math:`C_{in}` 为输入通道数， :math:`j` 的范围从 :math:`0` 到 :math:`C_{out} - 1` ， :math:`W_{ij}` 对应第 :math:`j` 个过滤器的第 :math:`i` 个通道， :math:`out_{j}` 对应输出的第 :math:`j` 个通道。 :math:`W_{ij}` 为卷积核的切片，其shape为 :math:`(\text{kernel_size[0]},\text{kernel_size[1]})` ，其中 :math:`\text{kernel_size[0]}` 和 :math:`\text{kernel_size[1]}` 是卷积核的高度和宽度。完整卷积核的shape为 :math:`(C_{out}, C_{in} / \text{groups}, \text{kernel_size[0]}, \text{kernel_size[1]})` ，其中 `groups` 是在通道上分割输入 `input` 的组数。

    如果 `pad_mode` 设置为"valid"，则输出宽度为 :math:`\left \lfloor{1 + \frac{W_{in} + \text{padding[0]} - \text{kernel_size} - (\text{kernel_size} - 1) \times(\text{dilation} - 1)}{\text { stride }}} \right \rfloor` 。
    其中， :math:`dialtion` 为卷积核元素之间的间距， :math:`stride` 为移动步长， :math:`padding` 为添加到输入两侧的零填充。
    对于取其他值的 `pad_mode` 时候的输出高度和宽度的计算，请参考 :class:`mindspore.nn.Conv1d` 里的计算公式。

    请参考论文 `Gradient Based Learning Applied to Document Recognition <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_ 。更详细的介绍，参见： `ConvNets <http://cs231n.github.io/convolutional-networks/>`_ 。

    .. note::
        在Ascend平台上，目前只支持深度卷积场景下的分组卷积运算。也就是说，当 `groups>1` 的场景下，必须要满足 `C_{in}` = `C_{out}` = `groups` 的约束条件。

    参数：
        - **input** (Tensor) - shape为 :math:`(N, C_{in}, W_{in})` 的Tensor。
        - **weight** (Tensor) - shape为 :math:`(C_{out}, C_{in}, W_{kernel}})` ，则卷积核shape为 :math:`(W_{kernel})` 。
        - **bias** (Tensor) - 偏置Tensor，shape为 :math:`(C_{out})` 的Tensor。如果 `bias` 是None，将不会添加偏置。默认值：None。
        - **pad_mode** (str，可选) - 指定填充模式。取值为"same"，"valid"，或"pad"。默认值："valid"。

          - **same**: 输出的宽度与输入整除 `stride` 后的值相同。填充将被均匀地添加到两侧，剩余填充量将被添加到维度末端。若设置该模式，`padding` 的值必须为0。
          - **valid**: 在不填充的前提下返回有效计算所得的输出。不满足计算的多余像素会被丢弃。如果设置此模式，则 `padding` 的值必须为0。
          - **pad**: 对输入 `input` 进行填充。在输入上填充 `padding` 大小的0。如果设置此模式， `padding` 必须大于或等于0。

        - **padding** (Union(int, tuple[int])，可选) - 输入 `input` 的宽度方向上填充的数量。数据类型为int或包含1个int组成的tuple。表示宽度方向的 `padding` 数量（左右两边均为该值）。值必须大于等于0，默认值：0。
        - **stride** (Union(int, tuple[int])，可选) - 卷积核移动的步长，数据类型为int或1个int组成的tuple。表示在宽度方向的移动步长。默认值：1。
        - **dilation** (Union(int, tuple[int])，可选) - 卷积核元素间的间隔。数据类型为int或1个int组成的tuple。若 :math:`k > 1` ，则卷积核间隔 `k` 个元素进行采样。垂直和水平方向上的 `k` ，其取值范围为[1, W]。默认值：1。
        - **groups** (int，可选) - 将过滤器拆分为组。默认值：1。

    返回：
        Tensor，卷积后的值。shape为 :math:`(N, C_{out}, W_{out})` 。

    异常：
        - **TypeError** -  `stride` 、 `padding` 或 `dilation` 既不是int也不是tuple。
        - **TypeError** -  `groups` 不是int。
        - **TypeError** -  `bias` 不是Tensor。
        - **ValueError** - `bias` 的shape不是 :math:`(C_{out})` 。
        - **ValueError** - `stride` 或 `diation` 小于1。
        - **ValueError** - `pad_mode` 不是"same"、"valid"或"pad"。
        - **ValueError** - `padding` 是一个长度不等于4的tuple。
        - **ValueError** - `pad_mode` 不等于"pad"时，`padding` 大于0。