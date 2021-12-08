mindspore.nn.Conv2d
====================

.. py:class:: mindspore.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, pad_mode="same", padding=0, dilation=1, group=1, has_bias=False, weight_init="normal", bias_init="zeros", data_format="NCHW")

   二维卷积层。

   对输入Tensor进行二维卷积，该Tensor的常见shape为 :math:`(N, C_{in}, H_{in}, W_{in})`，其中 :math:`N` 为batch size，:math:`C_{in}` 为通道数，:math:`H_{in},W_{in}` 分别为高度和宽度。对于每个batch中的Tensor，其shape为 :math:`(C_{in}, H_{in}, W_{in})` ，二维卷积公式定义如下：

   .. math:: out_j = \sum_{i=0}^{C_{in} - 1} ccor(W_{ij}, X_i) + b_j,

   其中 :math:`corr` 是互关联算子，:math:`C_{in}` 是输入通道数目，:math:`j` 的范围在 :math:`[0，C_{out}-1]` 内，:math:`W_{ij}`对应第 :math:`j` 个过滤器的第 :math:`i` 个通道，:math:`out_j` 对应输出的第 :math:`j` 个通道。:math:`W_{ij}` 是shape为 :math:`(\text{kernel_size[0]}, \text{kernel_size[1]})` 的kernel切片。其中 :math:`\text{kernel_size[0]}` 和 :math:`\text{kernel_size[1]}` 是卷积核的高度和宽度。完整kernel的shape为 :math:`(C_{out}, C_{in} // \text{group}, \text{kernel_size[0]}, \text{kernel_size[1]})`，其中group是在通道维度上分割输入 `x` 的组数。
   如果'pad_mode'被设置为 "valid"，输出高度和宽度分别为 :math:`\left \lfloor{1 + \frac{H_{in} + \text{padding[0]} + \text{padding[1]} - \text{kernel_size[0]} -
   (\text{kernel_size[0]} - 1) \times (\text{dilation[0]} - 1) }{\text{stride[0]}}} \right \rfloor` 和 :math:`\left \lfloor{1 + \frac{W_{in} + \text{padding[2]} + \text{padding[3]} - \text{kernel_size[1]} -
   (\text{kernel_size[1]} - 1) \times (\text{dilation[1]} - 1) }{\text{stride[1]}}} \right \rfloor`。

   详细介绍请参考论文 `Gradient Based Learning Applied to Document Recognition <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_ 。

   **参数：**

   - **in_channels** (`int`) – 输入的通道数 :math:`C_{in}` 。
   - **out_channels** (`dict`) - 输出的通道数 :math:`C_{out}` 。
   - **kernel_size** (`Union[int, tuple[int]]`) – 指定二维卷积窗口的高度和宽度。数据类型为整型或2个整型的tuple。一个整数表示卷积核的高度和宽度均为该值。2个整数的tuple分别表示卷积核的高度和宽度。
   - **stride** (`Union[int, tuple[int]]`) – 步长大小。数据类型为整型或2个整型的tuple。一个整数表示在高度和宽度方向的滑动步长均为该值。2个整数的tuple分别表示在高度和宽度方向的滑动步长。默认值：1。

      - **pad_mode** (`str`) – 指定填充模式。可选值为“same”，“valid”，“pad”。默认值：“same”。

         - **same**：采用补全方式。输出的高度和宽度与输入 `x` 一致。填充总数将在水平和垂直方向进行计算。并尽可能均匀分布到顶部、底部、左侧和右侧。否则，最后一次将从底部和右侧进行额外的填充。若设置该模式，`padding` 必须为0。
         - **valid**：采用丢弃方式。在不填充的前提下返回可能大的高度和宽度的输出。多余的像素会被丢弃。若设置该模式，`padding` 必须为0。
         - **pad**：输入 `x` 两侧的隐式填充。`padding` 的数量将填充到输入Tensor边框上。`padding` 必须大于或等于0。

      - **padding** (`Union[int, tuple[int]]`) –  输入 `x` 两侧的隐式填充。数据类型为int或包含4个整数的tuple。如果 `padding` 是一个整数，那么上、下、左、右的填充都等于 `padding` 。如果 `padding` 是一个有4个整数的tuple，那么上、下、左、右的填充分别等于 `padding[0]` 、 `padding[1]` 、 `padding[2]` 和 `padding[3]` 。默认值：0。
      - **dilation** (`Union[int, tuple[int]]`) –  指定用于扩张卷积的扩张速率。数据类型为整型或具有2个整型的tuple。如果设置 :math:`k > 1` ，则每个采样位置将跳过 :math:`k-1` 个像素。其值必须大于或等于1，并以输入的高度和宽度为边界。默认值：1。
      - **group** (`int`) –  将过滤器分组， `in_channels` 和 `out_channels` 必须被组数整除。如果组数等于 `in_channels` 和 `out_channels` ，这个二维卷积层也被称为二维深度卷积层。默认值：1.
      - **has_bias** (`bool`) –  指定图层是否使用偏置向量。默认值：False。
      - **weight_init** (`Union[Tensor, str, Initializer, numbers.Number]`) – 卷积核的初始化方法。它可以是Tensor，str，初始化实例或numbers.Number。当使用str时，可选“TruncatedNormal”，“Normal”，“Uniform”，“HeUniform”和“XavierUniform”分布以及常量“One”和“Zero”分布的值，可接受别名“ xavier_uniform”，“ he_uniform”，“ ones”和“ zeros”。上述字符串大小写均可。更多细节请参考Initializer的值。默认值：“normal”。
      - **bias_init** (`Union[Tensor, str, Initializer, numbers.Number]`) – 偏置向量的初始化方法。可以使用的初始化方法和字符串与“weight_init”相同。更多细节请参考Initializer的值。默认值：“zeros”。
      - **data_format** (`str`) –  数据格式的可选值有“NHWC”，“NCHW”。默认值：“NCHW”。

   **输入：**

   - **x** (Tensor) - Shape为 :math:`(N, C_{in}, H_{in}, W_{in})` 或者 :math:`(N, H_{in}, W_{in}, C_{in})` 的Tensor。

   **输出：**

   Tensor，shape为 :math:`(N, C_{out}, H_{out}, W_{out})` 或者 :math:`(N, H_{out}, W_{out}, C_{out})` 。

   **异常：**

   - **TypeError** - 如果 `in_channels` ， `out_channels` 或者 `group` 不是整数。
   - **TypeError** - 如果 `kernel_size` ， `stride`，`padding` 或者 `dilation` 既不是整数也不是tuple。
   - **ValueError** - 如果 `in_channels` ， `out_channels`，`kernel_size` ， `stride` 或者 `dilation` 小于1。
   - **ValueError** - 如果 `padding` 小于0。
   - **ValueError** - 如果 `pad_mode` 不是“same”，“valid”，“pad”中的一个。
   - **ValueError** - 如果 `padding` 是一个长度不等于4的tuple。
   - **ValueError** - 如果 `pad_mode` 不等于“pad”且 `padding` 不等于(0,0,0,0)。
   - **ValueError** - 如果 `data_format` 既不是“NCHW”也不是“NHWC”。


   **支持平台：**

   ``Ascend`` ``GPU`` ``CPU``

   **样例：** 

   >>> net = nn.Conv2d(120, 240, 4, has_bias=False, weight_init='normal')
   >>> x = Tensor(np.ones([1, 120, 1024, 640]), mindspore.float32)
   >>> output = net(x).shape
   >>> print(output)
   (1, 240, 1024, 640)