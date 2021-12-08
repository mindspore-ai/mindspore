mindspore.nn.Conv3dTranspose
=============================

.. py:class:: mindspore.nn.Conv3dTranspose(in_channels, out_channels, kernel_size, stride=1, pad_mode='same', padding=0, dilation=1, group=1, output_padding=0, has_bias=False, weight_init='normal', bias_init='zeros', data_format='NCDHW')

    三维转置卷积层。

    计算三维转置卷积，也称为反卷积（实际不是实际的反卷积）。
    
    转置卷积算子将每个输入元素乘以learnable kernel，并把所有输入特征平面的输出相加。
    
    该算子可以看成Conv3d相对于其输入的梯度。

    `x` 通常shape为 :math:`(N, C, D, H, W)` ，其中 :math:`N` 是batch size， :math:`C` 是通道数， :math:`D` 是特征层的深度， :math:`H` 是特征高度， :math:`W` 是特征层的宽度。
    
    转置卷积的计算过程相当于卷积的反向计算。

    对输入填充 :math:`dilation * (kernel\_size - 1) - padding` 个零。
    因此，当Conv3d和ConvTranspose3d使用相同的参数初始化时，它们的输入和输出shape是互逆的。
    但是，当stride>1时，Conv3d将多个输入的shape映射到同一个输出shape。
    ConvTranspose3d提供padding参数，增加一侧或多侧计算的输出shape。

    输出的高度和宽度定义如下：

    如果'pad_mode'设置为"pad"，

    .. math::
        D_{out} = (D_{in} - 1) \times \text{stride_d} - 2 \times \text{padding_d} + \text{dilation_d} \times
        (\text{kernel_size_d} - 1) + \text{output_padding_d} + 1

        H_{out} = (H_{in} - 1) \times \text{stride_h} - 2 \times \text{padding_h} + \text{dilation_h} \times
        (\text{kernel_size_h} - 1) + \text{output_padding_h} + 1

        W_{out} = (W_{in} - 1) \times \text{stride_w} - 2 \times \text{padding_w} + \text{dilation_w} \times
        (\text{kernel_size_w} - 1) + \text{output_padding_w} + 1

    如果'pad_mode'设置为"same"，

    .. math::

        D_{out} = (D_{in} + \text{stride_d} - 1)/\text{stride_d} \\
        H_{out} = (H_{in} + \text{stride_h} - 1)/\text{stride_h} \\
        W_{out} = (W_{in} + \text{stride_w} - 1)/\text{stride_w}

    如果'pad_mode'设置为"valid"，

    .. math::

        D_{out} = (D_{in} - 1) \times \text{stride_d} + \text{dilation_d} \times
        (\text{kernel_size_d} - 1) + 1 \\
        H_{out} = (H_{in} - 1) \times \text{stride_h} + \text{dilation_h} \times
        (\text{kernel_size_h} - 1) + 1 \\
        W_{out} = (W_{in} - 1) \times \text{stride_w} + \text{dilation_w} \times
        (\text{kernel_size_w} - 1) + 1

    **参数：**

    - **in_channels** (int) - 输入通道数 :math:`C_{in}` 。
    - **out_channels** (int) - 输出通道数 :math:`C_{out}` 。
    - **kernel_size** (Union[int, tuple[int]]) - 指定三维卷积窗口的深度、高度和宽度。数据类型为int或包含3个整数的tuple。一个整数表示卷积核的深度、高度和宽度均为该值该值。包含3个整数的tuple分别表示卷积核的深度、高度和宽度。
    - **stride** (Union[int, tuple[int]]) - 步长大小。数据类型为整型或3个整型的tuple。一个整数表示在深度、高度和宽度方向的滑动步长均为该值。3个整数的tuple分别表示在深度、高度和宽度方向的滑动步长。必须大于等于1。默认值：1。
    - **pad_mode** (str) - 选择填充模式。可选值为"pad"、"same"、"valid"。默认值："same"。

      - same：采用补全方式。输出的深度、高度和宽度与输入 `x` 一致。填充总数将在深度、水平和垂直方向进行计算。并尽可能均匀分布到头部、尾部、顶部、底部、左侧和右侧。否则，最后一次将从尾部、底部和右侧进行额外的填充。若设置该模式， `padding` 和 `output_padding` 必须为0。
      - valid：采用的丢弃方式。在不填充的前提下返回可能大的深度、高度和宽度的输出。多余的像素会被丢弃。若设置该模式， `padding` 和 `output_padding` 必须为0。
      - pad：输入 `x` 两侧的隐式填充。 `padding` 的数量将填充到输入Tensor边框上。 `padding` 必须大于或等于0。

    - **padding** (Union(int, tuple[int])) - 待填充的padding值。数据类型为int或包含6个整数的tuple。如果 `padding` 是一个整数，则头部、尾部、顶部，底部，左边和右边的填充都等于 `padding` 。如果 `padding` 是6个整数的tuple，则头部、尾部、顶部、底部、左边和右边的填充分别等于填充padding[0]、padding[1]、padding[2]、padding[3]、padding[4]和padding[5]。默认值：0。 
    - **dilation** (Union(int, tuple[int])) - 指定用于扩张卷积的扩张速率。数据类型为int或3个整数的tuple。:math:`(dilation_d, dilation_h, dilation_w)` 。目前，深度扩张仅支持1个用例的情况。如果设置为 :math:`k > 1` ，则每个采样位置都跳过 :math:`k - 1` 个像素。其值必须大于或等于1，并以输入 `x` 的深度、高度和宽度为界。默认值：1。
    - **group** (int) - 将过滤器拆分为组， `in_ channels` 和 `out_channels` 必须可被组数整除。默认值：1。当前仅支持1个。
    - **output_padding** (Union(int, tuple[int])) - 为输出的每个维度添加额外的大小。默认值：0。必须大于或等于0。
    - **has_bias** (bool) - 指定图层是否使用偏置矢量。默认值：False。
    - **weight_init** (Union[Tensor, str, Initializer, numbers.Number]) - 卷积核的初始化方法。它可以是Tensor，str，初始化实例或numbers.Number。当使用str时，可选“TruncatedNormal”，“Normal”，“Uniform”，“HeUniform”和“XavierUniform”分布以及常量“One”和“Zero”分布的值，可接受别名“ xavier_uniform”，“ he_uniform”，“ ones”和“ zeros”。上述字符串大小写均可。更多细节请参考Initializer的值。默认值：“normal”。
    - **bias_init** (Union[Tensor, str, Initializer, numbers.Number]) - 偏置向量的初始化方法。可以使用的初始化方法和字符串与“weight_init”相同。更多细节请参考Initializer的值。默认值：“zeros”。
    - **data_format** (str) - 数据格式的可选值。目前仅支持'NCDHW'。

    **输入：**

    - **x** (Tensor) - shape为 :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` 的Tensor。目前输入数据类型只支持float16和float32。

    **输出：**

    Tensor，shape为 :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` 。

    **支持平台：**

    ``Ascend`` ``GPU``

    **异常：**

    - **TypeError** - `in_channels` 、 `out_channels` 或 `group` 不是int。
    - **TypeError** - `kernel_size` 、 `stride` 、 `padding` 、 `dilation` 或 `output_padding` 既不是int也不是tuple。     
    - **TypeError** - 输入数据类型不是float16或float32。
    - **ValueError** - `in_channels` 、 `out_channels` 、 `kernel_size` 、 `stride` 或 `dilation` 小于1。
    - **ValueError** - `padding` 小于0。
    - **ValueError** - `pad_mode` 不是“same”，“valid”或“pad”。
    - **ValueError** - `padding` 是长度不等于6的tuple。
    - **ValueError** - `pad_mode` 不等于'pad'且 `padding` 不等于(0, 0, 0, 0, 0, 0)。
    - **ValueError** - `data_format` 不是'NCDHW'。

    **样例：**

    >>> x = Tensor(np.ones([32, 16, 10, 32, 32]), mindspore.float32)
    >>> conv3d_transpose = nn.Conv3dTranspose(in_channels=16, out_channels=3, kernel_size=(4, 6, 2),
    ...                                       pad_mode='pad')
    >>> output = conv3d_transpose(x)
    >>> print(output.shape)
    (32, 3, 13, 37, 33)
    