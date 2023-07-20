mindspore.ops.Conv3DTranspose
=============================

.. py:class:: mindspore.ops.Conv3DTranspose(in_channel, out_channel, kernel_size, mode=1, stride=1, pad_mode='valid', pad=0, dilation=1, group=1, output_padding=0, data_format='NCDHW')

    计算三维转置卷积，也称为反卷积（实际不是真正的反卷积）。

    输入的shape通常为 :math:`(N, C, D, H, W)`, 其中 :math:`N` 为batch size，:math:`C` 是空间维度，:math:`D` 、 :math:`H` 和 :math:`W` 分别为特征层的深度、高度和宽度。

    若 `pad_mode` 被设定为 "pad"，则输出的深度，高度和宽度被定义为：

    .. math::
        D_{out} = (D_{in} - 1) \times \text{stride}[0] - 2 \times \text{pad}[0] + \text{dilation}[0]
        \times (\text{kernel_size}[0] - 1) + \text{output_padding}[0] + 1

        H_{out} = (H_{in} - 1) \times \text{stride}[1] - 2 \times \text{pad}[1] + \text{dilation}[1]
        \times (\text{kernel_size}[1] - 1) + \text{output_padding}[1] + 1

        W_{out} = (W_{in} - 1) \times \text{stride}[2] - 2 \times \text{pad}[2] + \text{dilation}[2]
        \times (\text{kernel_size}[2] - 1) + \text{output_padding}[2] + 1

    .. note::
        在Ascend平台上，目前只支持 :math:`group=1` 。

    参数：
        - **in_channel** (int) - 输入 `dout` 的通道数。
        - **out_channel** (int) - 输入 `weight` 的通道数。
        - **kernel_size** (Union[int, tuple[int]]) - 指定三维卷积核的深度、高度和宽度。数据类型为int或包含三个int值的Tuple。为int时表示卷积核的深度、高度和宽度均为该值。包含三个int值的Tuple分别表示卷积核的深度、高度和宽度。
        - **mode** (int，可选) - 指定不同的卷积模式。此值目前未被使用。默认值： ``1`` 。
        - **pad_mode** (str，可选) - 指定填充模式，填充值为0。可选值为 ``"same"`` ， ``"valid"`` 或 ``"pad"`` 。默认值： ``"valid"`` 。

          - ``"same"``：在输入的深度、高度和宽度维度进行填充，使得当 `stride` 为 ``1`` 时，输入和输出的shape一致。待填充的量由算子内部计算，若为偶数，则均匀地填充在四周，若为奇数，多余的填充量将补充在前方/底部/右侧。如果设置了此模式， `pad` 必须为0。
          - ``"valid"``：不对输入进行填充，返回输出可能的最大深度、高度和宽度，不能构成一个完整stride的额外的像素将被丢弃。如果设置了此模式， `pad` 必须为0。
          - ``"pad"``：对输入填充指定的量。在这种模式下，在输入的深度、高度和宽度方向上填充的量由 `pad` 参数指定。如果设置此模式， `pad` 必须大于或等于0。
        
        - **pad** (Union(int, tuple[int])，可选) - 在输入各维度两侧填充的数量。如果 `pad` 是一个整数，则前部、后部、顶部，底部，左边和右边的填充都等于 `pad` 。如果 `pad` 是6个整数的Tuple，则前部、后部、顶部、底部、左边和右边的填充分别等于填充 `pad[0]` 、 `pad[1]` 、 `pad[2]` 、 `pad[3]` 、 `pad[4]` 和 `pad[5]` 。默认值： ``0`` 。
        - **stride** (Union(int, tuple[int])，可选) - 三维卷积核的移动步长。数据类型为整型或三个整型的Tuple。一个整数表示在深度、高度和宽度方向的移动步长均为该值。三个整数的Tuple分别表示在深度、高度和宽度方向的移动步长。默认值： ``1`` 。
        - **dilation** (Union(int, tuple[int])，可选) - 卷积核膨胀尺寸，指定应用卷积核的间隔。默认值： ``1`` 。
        - **group** (int，可选) - 将过滤器拆分的组数， `in_channels` 和 `out_channels` 必须可被 `group` 整除。默认值： ``1`` 。
        - **output_padding** (Union(int, tuple[int])，可选) - 为输出的各个维度添加额外长度。默认值： ``0`` 。
        - **data_format** (str，可选) - 支持的数据模式。目前仅支持 ``"NCDHW"`` 。默认值： ``"NCDHW"`` 。

    输入：
        - **dout** (Tensor) - 卷积操作的输出的梯度Tensor。shape： :math:`(N, C_{in}, D_{out}, H_{out}, W_{out})` 。目前数据类型仅支持float16和float32。
        - **weight** (Tensor) - 若kernel shape为 :math:`(K_d, K_h, K_w)` ，则weight shape应为 :math:`(C_{in}, C_{out}//group, K_d, K_h, K_w)` ，其中 :math:`group` 为算子参数。:math:`//` 为整数除法操作。目前数据类型仅支持float16和float32。
        - **bias** (Tensor) - shape为 :math:`C_{out}` 的Tensor。目前仅支持 ``None`` 。默认值： ``None`` 。

    输出：
        卷积操作的输入的梯度Tensor，shape： :math:`(N, C_{out}//group, D_{out}, H_{out}, W_{out})` ，其中 :math:`group` 为算子参数。

    异常：
        - **TypeError** - `in_channel` 、 `out_channel` 或 `group` 不是int。
        - **TypeError** - `kernel_size` 、 `stride` 、 `pad` 、 `dilation` 或 `output_padding` 既不是int也不是Tuple。
        - **ValueError** - `in_channel` 、 `out_channel` 、 `kernel_size` 、 `stride` 或 `dilation` 小于1。
        - **ValueError** - `pad` 小于0。
        - **ValueError** - `pad_mode` 取值非"same"、"valid"或"pad"。
        - **ValueError** - `pad` 为长度不等于6的Tuple。
        - **ValueError** - `pad_mode` 未设定为"pad"且 `pad` 不等于(0, 0, 0, 0, 0, 0)。
        - **ValueError** - `data_format` 取值非"NCDHW"。
        - **TypeError** - `dout` 或 `weight` 的数据类型不是float16。
        - **ValueError** - `bias` 不为None。 `dout` 或 `weight` 的秩不为5。
