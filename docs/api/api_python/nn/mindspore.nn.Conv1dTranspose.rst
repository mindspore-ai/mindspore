mindspore.nn.Conv1dTranspose
=============================

.. py:class:: mindspore.nn.Conv1dTranspose(in_channels, out_channels, kernel_size, stride=1, pad_mode='same', padding=0, dilation=1, group=1, has_bias=False, weight_init=None, bias_init=None, dtype=mstype.float32)

    计算一维转置卷积，可以视为Conv1d对输入求梯度，也称为反卷积（实际不是真正的反卷积）。

    输入的shape通常是 :math:`(N, C_{in}, L_{in})` ，其中 :math:`N` 是batch size， :math:`C_{in}` 是通道数， :math:`L_{in}` 是序列的长度。
    当Conv1d和ConvTranspose1d使用相同的参数初始化时，且 `pad_mode` 设置为"pad"，它们会在输入的两端填充 :math:`dilation * (kernel\_size - 1) - padding` 个零，这种情况下它们的输入和输出shape是互逆的。
    然而，当 `stride` 大于1时，Conv1d会将多个输入的shape映射到同一个输出shape。反卷积网络可以参考 `Deconvolutional Networks <https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf>`_ 。

    参数：
        - **in_channels** (int) - Conv1dTranspose层输入Tensor的空间维度。
        - **out_channels** (int) - Conv1dTranspose层输出Tensor的空间维度。
        - **kernel_size** (int) - 指定一维卷积核的宽度。
        - **stride** (int) - 一维卷积核的移动步长，默认值：``1`` 。
        - **pad_mode** (str，可选) - 指定填充模式，填充值为0。可选值为 ``"same"`` ， ``"valid"`` 或 ``"pad"`` 。默认值： ``"same"`` 。

          - ``"same"``：在输入的两端填充，使得当 `stride` 为 ``1`` 时，输入和输出的shape一致。待填充的量由算子内部计算，若为偶数，则均匀地填充在四周，若为奇数，多余的填充量将补充在右端。如果设置了此模式， `padding` 必须为0。
          - ``"valid"``：不对输入进行填充，返回输出可能的最大长度，不能构成一个完整stride的额外的像素将被丢弃。如果设置了此模式， `padding` 必须为0。
          - ``"pad"``：对输入填充指定的量。在这种模式下，填充的量由 `padding` 参数指定。如果设置此模式， `padding` 必须大于或等于0。

        - **padding** (int) - 输入两侧填充的数量。默认值： ``0`` 。
        - **dilation** (int) - 一维卷积核膨胀尺寸。若 :math:`k > 1` ，则kernel间隔 `k` 个元素进行采样。 `k` 取值范围为[1, L]。默认值：``1`` 。
        - **group** (int) - 将过滤器拆分为组， `in_channels` 和 `out_channels` 必须可被 `group` 整除。当 `group` 大于1时，暂不支持Ascend平台。默认值：``1`` 。
        - **has_bias** (bool) - Conv1dTranspose层是否添加偏置参数。默认值： ``False`` 。
        - **weight_init** (Union[Tensor, str, Initializer, numbers.Number]) - 权重参数的初始化方法。它可以是Tensor，str，Initializer或numbers.Number。当使用str时，可选 ``"TruncatedNormal"`` ， ``"Normal"`` ， ``"Uniform"`` ， ``"HeUniform"`` 和 ``"XavierUniform"`` 分布以及常量 ``"One"`` 和 ``"Zero"`` 分布的值，可接受别名 ``"xavier_uniform"`` ， ``"he_uniform"`` ， ``"ones"`` 和 ``"zeros"`` 。上述字符串大小写均可。更多细节请参考Initializer的值。默认值： ``None`` ，权重使用HeUniform初始化。
        - **bias_init** (Union[Tensor, str, Initializer, numbers.Number]) - 偏置参数的初始化方法。可以使用的初始化方法与"weight_init"相同。更多细节请参考Initializer的值。默认值： ``None`` ，偏差使用Uniform初始化。
        - **dtype** (:class:`mindspore.dtype`) - Parameters的dtype。默认值： ``mstype.float32`` 。

    输入：
        - **x** (Tensor) - shape为 :math:`(N, C_{in}, L_{in})` 的Tensor。

    输出：
        Tensor，shape为 :math:`(N, C_{out}, L_{out})` 。

        - 当 `pad_mode` 设置为 ``"same"`` 时：
          :math:`L_{out} = \frac{ L_{in} + \text{stride} - 1 }{ \text{stride} }`
        - 当 `pad_mode` 设置为 ``"valid"`` 时：
          :math:`L_{out} = (L_{in} - 1) \times \text{stride} + \text{dilation} \times (\text{kernel_size} - 1) + 1`
        - 当 `pad_mode` 设置为 ``"pad"`` 时：
          :math:`L_{out} = (L_{in} - 1) \times \text{stride} - 2 \times \text{padding} + \text{dilation} \times (\text{kernel_size} - 1) + 1`

    异常：
        - **TypeError** - `in_channels` 、 `out_channels` 、 `kernel_size` 、 `stride` 、 `padding` 或 `dilation` 不是int。
        - **ValueError** - `in_channels` 、 `out_channels` 、 `kernel_size` 、 `stride` 或 `dilation` 小于1。
        - **ValueError** - `padding` 小于0。
        - **ValueError** - `pad_mode` 不是 ``"same"`` ， ``"valid"`` 或 ``"pad"`` 。
