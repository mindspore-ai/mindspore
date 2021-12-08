mindspore.nn.Conv1d
======================

.. py:class:: mindspore.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, pad_mode='same', padding=0, dilation=1, group=1, has_bias=False, weight_init='normal', bias_init='zeros')

    一维卷积层。

    在输入Tensor上应用一维卷积，该Tensor的shape通常为 :math:`(N, C_{in}, W_{in})` ，其中 :math:`N` 是batch size， :math:`C_{in}` 是通道数。
    对于每个batch中的Tensor，其shape为 :math:`(C_{in}, W_{in})` ，公式定义为：

    .. math::

        out_j = \sum_{i=0}^{C_{in} - 1} ccor(W_{ij}, X_i) + b_j,

    其中， :math:`ccor` 为互关联算子， :math:`C_{in}` 为输入通道数， :math:`j` 的范围从 :math:`0` 到 :math:`C_{out} - 1` ， :math:`W_{ij}` 对应第 :math:`j` 个过滤器的第 :math:`i` 个通道， :math:`out_{j}` 对应输出的第 :math:`j` 个通道。
    :math:`W_{ij}`是kernel的切片，它的shape为 :math:`(\text{ks_w})` ，其中 :math:`\text{ks_w}` 是卷积核的宽度。
    完整kernel的shape为 :math:`(C_{out}, C_{in} // \text{group}, \text{ks_w})` ，其中group是在通道维度上分割输入 `x` 的组数。
    如果'pad_mode'设置为"valid"，则输出宽度将为 :math:`\left \lfloor{1 + \frac{W_{in} + 2 \times \text{padding} - \text{ks_w} - (\text{ks_w} - 1) \times (\text{dilation} - 1) }{\text{stride}}} \right \rfloor` 。
    论文 `Gradient Based Learning Applied to Document Recognition <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_ 首次提出卷积层。
    
    **参数：**
    
    - **in_channels** (int) - 输入通道数 :math:`C_{in}` 。
    - **out_channels** (int) - 输出通道数 :math:`C_{out}` 。
    - **kernel_size** (int) - 指定一维卷积核的宽度。
    - **stride** (int) - 步长大小，表示移到宽度。默认值：1。
    - **pad_mode** (str) - 指定填充模式。可选值为"same"、"valid"、"pad"。默认值："same"。

      - same：采用补全方式。输出的宽度与输入 `x` 一致。填充总数将在水平上进行计算。并尽可能均匀分布到左侧和右侧。否则，最后一次将从底部和右侧进行额外的填充。若设置该模式，`padding` 必须为0。
      - valid：采用的丢弃方式。在不填充的前提下返回可能大的宽度的输出。多余的像素会被丢弃。如果设置此模式，则 `padding` 必须为0。
      - pad：输入 `x` 两侧的隐式填充。 `padding` 的数量将填充到输入Tensor边框上。 `padding` 必须大于或等于0。

    - **padding** (int) - 输入 `x` 两侧的隐式填充。默认值：0。
    - **dilation** (int) - 指定用于扩张卷积的扩张速率。如果设置为 :math:`k > 1` ，则每个采样位置都跳过 :math:`k - 1` 个像素。其值必须大于或等于1，并以输入 `x` 的宽度为界。默认值：1。
    - **group** (int) - 将过滤器拆分为组， `in_ channels` 和 `out_channels` 必须可被组数整除。默认值：1。
    - **has_bias** (bool) - 指定图层是否使用偏置矢量。默认值：False。
    - **weight_init** (Union[Tensor, str, Initializer, numbers.Number]) - 卷积核的初始化方法。它可以是Tensor，str，初始化实例或numbers.Number。当使用str时，可选“TruncatedNormal”，“Normal”，“Uniform”，“HeUniform”和“XavierUniform”分布以及常量“One”和“Zero”分布的值，可接受别名“ xavier_uniform”，“ he_uniform”，“ ones”和“ zeros”。上述字符串大小写均可。更多细节请参考Initializer的值。默认值：“normal”。
    - **bias_init** (Union[Tensor, str, Initializer, numbers.Number]) - 偏置向量的初始化方法。可以使用的初始化方法和字符串与“weight_init”相同。更多细节请参考Initializer的值。默认值：“zeros”。

    **输入：**
    
    - **x** (Tensor) - shape为 :math:`(N, C_{in}, W_{in})` 的Tensor。

    **输出：**
    
    Tensor，shape为:math:`(N, C_{out}, W_{out})`。

    **异常：**

    - **TypeError** - `in_channels` 、 `out_channels` 、 `kernel_size` 、 `stride` 、 `padding` 或 `dilation` 不是int。
    - **ValueError** - `in_channels` 、 `out_channels` 、 `kernel_size` 、 `stride` 或 `dilation` 小于1。
    - **ValueError** - `padding` 小于0。
    - **ValueError** - `pad_mode` 不是'same'，'valid'，或'pad'。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> net = nn.Conv1d(120, 240, 4, has_bias=False, weight_init='normal')
    >>> x = Tensor(np.ones([1, 120, 640]), mindspore.float32)
    >>> output = net(x).shape
    >>> print(output)
    (1, 240, 640)
    