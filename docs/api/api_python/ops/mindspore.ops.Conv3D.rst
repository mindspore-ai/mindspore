mindspore.ops.Conv3D
====================

.. py:class:: mindspore.ops.Conv3D(out_channel, kernel_size, mode=1, pad_mode="valid", pad=0, stride=1, dilation=1, group=1, data_format="NCDHW")

    三维卷积操作。

    对输入Tensor进行3维卷积操作。输入Tensor的shape通常为 :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` ，输出的shape为 :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` ，其中 :math:`N` 为batch size，:math:`C` 是通道数， :math:`D` 、 :math:`H` 、 :math:`W` 分别为特征层的深度、高度和宽度。公式定义如下：

    .. math::
        \operatorname{out}\left(N_{i}, C_{\text {out}_j}\right)=\operatorname{bias}\left(C_{\text {out}_j}\right)+
        \sum_{k=0}^{C_{in}-1} ccor(\text {weight}\left(C_{\text {out}_j}, k\right),
        \operatorname{input}\left(N_{i}, k\right))

    其中 :math:`k` 为kernel， :math:`ccor` 是 `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_ 。

    如果指定 `pad_mode` 为 "valid"，则输出的深度、高度和宽度分别为
    :math:`\left \lfloor{1 + \frac{D_{in} + 2 \times \text{padding} - \text{ks_d} -
    (\text{ks_d} - 1) \times (\text{dilation} - 1) }{\text{stride}}} \right \rfloor` 、
    :math:`\left \lfloor{1 + \frac{H_{in} + 2 \times \text{padding} - \text{ks_h} -
    (\text{ks_h} - 1) \times (\text{dilation} - 1) }{\text{stride}}} \right \rfloor` 以及 
    :math:`\left \lfloor{1 + \frac{W_{in} + 2 \times \text{padding} - \text{ks_w} -
    (\text{ks_w} - 1) \times (\text{dilation} - 1) }{\text{stride}}} \right \rfloor` ，其中
    :math:`dilation` 为三维卷积核膨胀尺寸， :math:`stride` 为移动步长，
    :math:`padding` 为在输入两侧的填充长度。

    参数：
        - **out_channel** (int) - 输出的通道数 :math:`C_{out}` 。
        - **kernel_size** (Union[int, tuple[int]]) - 指定三维卷积核的深度、高度和宽度。数据类型为int或包含三个整数的Tuple。一个整数表示卷积核的深度、高度和宽度均为该值。包含三个整数的Tuple分别表示卷积核的深度、高度和宽度。
        - **mode** (int) - 指定不同的卷积模式。此值目前未被使用。默认值: 1。
        - **stride** (Union[int, tuple[int]]) - 当 `stride` 为int时表示在深度、高度和宽度方向的移动步长均为该值。当 `stride` 为三个int值所组成的Tuple时，三个int值分别表示在深度、高度和宽度方向的移动步长。默认值：1。
        - **pad_mode** (str) - 指定填充模式。可选值为"same"、"valid"、"pad"。默认值："valid"。

          - same: 输出的深度、高度和宽度分别与对应输入整除 `stride` 后的值相同。
            填充将被均匀地添加到高和宽的两侧，剩余填充量将被添加到维度末端。
            若设置该模式， `pad` 的值必须为0。

          - valid: 在不填充的前提下返回有效计算所得的输出。不满足计算的多余像素会被丢弃。如果设置此模式，则 `pad` 的值必须为0。

          - pad: 在输入深度、高度和宽度各维度两侧添加 `pad` 数量的填充。如果设置此模式， `pad` 的值必须大于或等于0。

        - **pad** (Union(int, tuple[int])) - 在输入各维度两侧的填充长度。如果 `pad` 是一个整数，则前部、后部、顶部，底部，左边和右边的填充都等于 `pad` 。如果 `pad` 是6个整数的Tuple，则前部、后部、顶部、底部、左边和右边的填充分别等于填充 `pad[0]` 、 `pad[1]` 、 `pad[2]` 、 `pad[3]` 、 `pad[4]` 和 `pad[5]` 。默认值：0。
        - **dilation** (Union[int, tuple[int]]) - 三维卷积核膨胀尺寸。数据类型为int或三个整数的Tuple :math:`(dilation_d, dilation_h, dilation_w)` 。目前在深度维度仅支持取值为1。若 :math:`k > 1` ，则kernel间隔 `k` 个元素取样。取值大于等于1且小于对应的高度或宽度大小。默认值: 1。
        - **group** (int) - 将过滤器拆分为组， `in_channels` 和 `out_channels` 必须可被 `group` 整除。默认值: 1。目前仅支持取值为1。
        - **data_format** (str) - 支持的数据模式。目前仅支持"NCDHW"。

    输入：
        - **x** (Tensor) - shape为 :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` 的Tensor。目前数据类型仅支持float16和float32。
        - **weight** (Tensor) - 若kernel shape为 :math:`(k_d, K_h, K_w)` ，则weight shape应为 :math:`(C_{out}, C_{in}/groups, k_d, K_h, K_w)` 。目前数据类型仅支持float16和float32。
        - **bias** (Tensor) - shape为 :math:`C_{in}` 的Tensor。目前仅支持None。默认值：None。

    输出：
        Tensor，shape为 :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` 。

    异常：
        - **TypeError** - `out_channel` 或 `group` 不是int。
        - **TypeError** - `kernel_size` 、 `stride` 、 `pad` 或 `dilation` 既不是int也不是Tuple。
        - **ValueError** - `out_channel` 、 `kernel_size` 、 `stride` 或 `dilation` 小于1。
        - **ValueError** - `pad` 小于0。
        - **ValueError** - `pad_mode` 取值非"same"、"valid"或"pad"。
        - **ValueError** - `pad` 为长度不等于6的Tuple。
        - **ValueError** - `pad_mode` 未设定为"pad"且 `pad` 不等于(0, 0, 0, 0, 0, 0)。
        - **ValueError** - `data_format` 取值非"NCDHW"。
