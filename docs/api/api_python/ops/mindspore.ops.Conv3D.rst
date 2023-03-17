mindspore.ops.Conv3D
====================

.. py:class:: mindspore.ops.Conv3D(out_channel, kernel_size, mode=1, pad_mode="valid", pad=0, stride=1, dilation=1, group=1, data_format="NCDHW")

    对输入Tensor计算三维卷积。该Tensor的常见shape为 :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` ，其中 :math:`N` 为batch size，:math:`C_{in}` 为通道数，:math:`D` 为深度， :math:`H_{in}, W_{in}` 分别为特征层的高度和宽度。 :math:`X_i` 为 :math:`i^{th}` 输入值， :math:`b_i` 为 :math:`i^{th}` 输入值的偏置项。对于每个batch中的Tensor，其shape为 :math:`(C_{in}, D_{in}, H_{in}, W_{in})` ，公式定义如下：

    .. math::
        \operatorname{out}\left(N_{i}, C_{\text {out}_j}\right)=\operatorname{bias}\left(C_{\text {out}_j}\right)+
        \sum_{k=0}^{C_{in}-1} ccor(\text {weight}\left(C_{\text {out}_j}, k\right),
        \operatorname{input}\left(N_{i}, k\right))

    其中，:math:`k` 为卷积核数，:math:`ccor` 为 `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_ ，
    :math:`C_{in}` 为输入通道数， :math:`j` 的范围从 :math:`0` 到 :math:`C_{out} - 1` ， :math:`W_{ij}` 对应第 :math:`j` 个过滤器的第 :math:`i` 个通道， :math:`out_{j}` 对应输出的第 :math:`j` 个通道。
    :math:`W_{ij}` 为卷积核的切片，其shape为 :math:`(\text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})` ，其中 :math:`\text{kernel_size[1]}` 和 :math:`\text{kernel_size[2]}` 是卷积核的高度和宽度，
    :math:`\text{kernel_size[0]}` 是卷积核的深度。完整卷积核的shape为 :math:`(C_{out}, C_{in} / \text{group}, \text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})` ，其中 `group` 是在通道上分割输入 `inputs` 的组数。

    详细内容请参考论文 `Gradient Based Learning Applied to Document Recognition <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_ 。

    .. note::
        在Ascend平台上，目前只支持深度卷积场景下的分组卷积运算。也就是说，当 `group>1` 的场景下，必须要满足 `C_{in}` = `C_{out}` = `group` 的约束条件。

    参数：
        - **out_channel** (int) - 输出的通道数 :math:`C_{out}` 。
        - **kernel_size** (Union[int, tuple[int]]) - 指定三维卷积核的深度、高度和宽度。数据类型为int或包含三个整数的Tuple。一个整数表示卷积核的深度、高度和宽度均为该值。包含三个整数的Tuple分别表示卷积核的深度、高度和宽度。
        - **mode** (int) - 指定不同的卷积模式。此值目前未被使用。默认值: 1。
        - **stride** (Union[int, tuple[int]]，可选) - 卷积核移动的步长，数据类型为int或两个int组成的tuple。一个int表示在高度和宽度方向的移动步长均为该值。两个int组成的tuple分别表示在高度和宽度方向的移动步长。默认值：1。
        - **pad_mode** (str，可选) - 指定填充模式。取值为"same"，"valid"，或"pad"。默认值："valid"。

          - **same**: 输出的高度和宽度分别与输入整除 `stride` 后的值相同。填充将被均匀地添加到高和宽的两侧，剩余填充量将被添加到维度末端。若设置该模式，`padding` 的值必须为0。
          - **valid**: 在不填充的前提下返回有效计算所得的输出。不满足计算的多余像素会被丢弃。如果设置此模式，则 `padding` 的值必须为0。
          - **pad**: 对输入 `inputs` 进行填充。在输入的高度和宽度方向上填充 `padding` 大小的0。如果设置此模式， `padding` 必须大于或等于0。

        - **pad** (Union(int, tuple[int])) - 在输入各维度两侧的填充长度。如果 `pad` 是一个整数，则前部、后部、顶部，底部，左边和右边的填充都等于 `pad` 。如果 `pad` 是6个整数的Tuple，则前部、后部、顶部、底部、左边和右边的填充分别等于填充 `pad[0]` 、 `pad[1]` 、 `pad[2]` 、 `pad[3]` 、 `pad[4]` 和 `pad[5]` 。默认值：0。
        - **dilation** (Union[int, tuple[int]]，可选) - 卷积核膨胀尺寸。数据类型为int或由3个int组成的tuple。若 :math:`k > 1` ，则卷积核间隔 `k` 个元素进行采样。前后、垂直和水平方向上的 `k` ，其取值范围分别为[1, D]、[1, H]和[1, W]。默认值：1。
        - **group** (int，可选) - 将过滤器拆分为组， `in_channels` 和 `out_channels` 必须可被 `group` 整除。默认值：1。
        - **data_format** (str) - 支持的数据模式。目前仅支持"NCDHW"。

    输入：
        - **x** (Tensor) - shape为 :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` 的Tensor。目前数据类型仅支持float16和float32。
        - **weight** (Tensor) - 若kernel shape为 :math:`(k_d, K_h, K_w)` ，则weight shape应为 :math:`(C_{out}, C_{in}/groups, k_d, K_h, K_w)` 。目前数据类型仅支持float16和float32。
        - **bias** (Tensor) - shape为 :math:`C_{in}` 的Tensor。目前仅支持None。默认值：None。

    输出：
        Tensor，卷积后的值。shape为 :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` 。

        `pad_mode` 为"same"时：

        .. math::
            \begin{array}{ll} \\
                D_{out} = \left \lceil{\frac{D_{in}}{\text{stride[0]}}} \right \rceil \\
                H_{out} = \left \lceil{\frac{H_{in}}{\text{stride[1]}}} \right \rceil \\
                W_{out} = \left \lceil{\frac{W_{in}}{\text{stride[2]}}} \right \rceil \\
            \end{array}

        `pad_mode` 为"valid"时：

        .. math::
            \begin{array}{ll} \\
                D_{out} = \left \lfloor{\frac{D_{in} - \text{dilation[0]} \times (\text{kernel_size[0]} - 1) }
                {\text{stride[0]}} + 1} \right \rfloor \\
                H_{out} = \left \lfloor{\frac{H_{in} - \text{dilation[1]} \times (\text{kernel_size[1]} - 1) }
                {\text{stride[1]}} + 1} \right \rfloor \\
                W_{out} = \left \lfloor{\frac{W_{in} - \text{dilation[2]} \times (\text{kernel_size[2]} - 1) }
                {\text{stride[2]}} + 1} \right \rfloor \\
            \end{array}

        `pad_mode` 为"pad"时：

        .. math::
            \begin{array}{ll} \\
                D_{out} = \left \lfloor{\frac{D_{in} + padding[0] + padding[1] - (\text{dilation[0]} - 1) \times
                \text{kernel_size[0]} - 1 }{\text{stride[0]}} + 1} \right \rfloor \\
                H_{out} = \left \lfloor{\frac{H_{in} + padding[2] + padding[3] - (\text{dilation[1]} - 1) \times
                \text{kernel_size[1]} - 1 }{\text{stride[1]}} + 1} \right \rfloor \\
                W_{out} = \left \lfloor{\frac{W_{in} + padding[4] + padding[5] - (\text{dilation[2]} - 1) \times
                \text{kernel_size[2]} - 1 }{\text{stride[2]}} + 1} \right \rfloor \\
            \end{array}

    异常：
        - **TypeError** - `out_channel` 或 `group` 不是int。
        - **TypeError** - `kernel_size` 、 `stride` 、 `pad` 或 `dilation` 既不是int也不是Tuple。
        - **ValueError** - `out_channel` 、 `kernel_size` 、 `stride` 或 `dilation` 小于1。
        - **ValueError** - `pad` 小于0。
        - **ValueError** - `pad_mode` 取值非"same"、"valid"或"pad"。
        - **ValueError** - `pad` 为长度不等于6的Tuple。
        - **ValueError** - `pad_mode` 未设定为"pad"且 `pad` 不等于(0, 0, 0, 0, 0, 0)。
        - **ValueError** - `data_format` 取值非"NCDHW"。
