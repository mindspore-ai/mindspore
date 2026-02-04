mindspore.mint.nn.Conv3d
=============================

.. py:class:: mindspore.mint.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', dtype=None)

    三维卷积层。

    对输入Tensor计算三维卷积。通常，输入Tensor的shape为 :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` ，其中 :math:`N` 为batch size，:math:`C` 为通道数，:math:`D, H, W` 分别为特征图的深度、高度和宽度。

    根据以下公式计算输出：

    .. math::

        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{ccor}({\text{weight}(C_{\text{out}_j}, k), \text{input}(N_i, k)})

    其中， :math:`bias` 为输出偏置，:math:`ccor` 为 `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_ 操作，
    :math:`weight` 为卷积核的值， :math:`input` 为输入的特征图。

    - :math:`i` 对应batch数，其范围为 :math:`[0, N-1]` ，其中 :math:`N` 为输入batch。

    - :math:`j` 对应输出通道，其范围为 :math:`[0, C_{out}-1]` ，其中 :math:`C_{out}` 为输出通道数，该值也等于卷积核的个数。

    - :math:`k` 对应输入通道数，其范围为 :math:`[0, C_{in}-1]`，其中 :math:`C_{in}` 为输入通道数，该值也等于卷积核的通道数。

    因此，上面的公式中， :math:`{bias}(C_{\text{out}_j})` 为第 :math:`j` 个输出通道的偏置， :math:`{weight}(C_{\text{out}_j}, k)` 表示第 :math:`j` 个
    卷积核在第 :math:`k` 个输入通道的卷积核切片， :math:`{input}(N_i, k)` 为特征图第 :math:`i` 个batch第 :math:`k` 个输入通道的切片。

    卷积核shape为 :math:`(\text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})` ，其中 :math:`\text{kernel_size[0]}` 、
    :math:`\text{kernel_size[1]}` 和 :math:`\text{kernel_size[2]}` 分别是卷积核的深度、高度和宽度。若考虑到输入输出通道以及 `groups` ，则完整卷积核的shape为
    :math:`(C_{out}, C_{in} / \text{groups}, \text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})` ，
    其中 `groups` 是分组卷积时在通道上分割输入 `input` 的组数。

    想更深入了解卷积层，请参考论文 `Gradient Based Learning Applied to Document Recognition <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_ 。

    参数的约束细节，请参考 :func:`mindspore.mint.nn.functional.conv3d` 。

    .. warning::
        仅支持 Atlas A2 训练系列产品。

    参数：
        - **in_channels** (int) - 输入图像中的通道数。
        - **out_channels** (int) - 卷积产生的通道数。
        - **kernel_size** (Union[int, tuple[int], list[int]]) - 指定三维卷积核的深度、高度和宽度。可以为单个int，或者由3个int组成的tuple/list。单个int表示卷积核的深度、高度和宽度均为该值；tuple/list中的3个int分别表示卷积核的深度、高度和宽度。
        - **stride** (Union[int, tuple[int], list[int]]，可选) - 三维卷积核的移动步长。可以为单个int，或者由3个int组成的tuple/list。单个int表示在深度、高度和宽度方向的移动步长均为该值；tuple/list中的3个int分别表示在深度、高度和宽度方向的移动步长。默认值： ``1`` 。
        - **padding** (Union[int, tuple[int], list[int], str]，可选) - 输入的深度、高度和宽度方向的填充数。可以为单个int、由3个int组成的tuple/list，或者string { ``"valid"`` , ``"same"`` }。该值应大于或等于0。默认值： ``0`` 。

          - ``"same"``：在输入的边缘加上衬垫，这样当 `stride` 设置为 ``1`` 时，输入和输出的形状是相同的。填充量由运算符内部计算。如果填充量是偶数，则均匀分布在输入周围；如果填充量为奇数，则多余的填充量会流向右侧/底部。
          - ``"valid"``：输入没有填充，输出返回最大可能的深度、高度和宽度。无法完成完整步幅的额外像素将被丢弃。

        - **dilation** (Union[int, tuple[int], list[int]]，可选) - 控制卷积核元素之间的间距。默认值： ``1`` 。
        - **groups** (int，可选) - 将过滤器拆分为组， `in_channels` 和 `out_channels` 必须可被 `groups` 整除。如果组数等于 `in_channels` 和 `out_channels` ，这个三维卷积层也被称为三维深度卷积层。默认值： ``1`` 。
          需要满足以下约束：

          - :math:`(C_{in} \text{ % } \text{groups} == 0)`
          - :math:`(C_{out} \text{ % } \text{groups} == 0)`
          - :math:`(C_{out} >= \text{groups})`
          - :math:`(\text{weight[1]} = C_{in} / \text{groups})`

        - **bias** (bool，可选) - Conv3d层是否具有偏置参数。默认值： ``True`` 。

        - **padding_mode** (str，可选) - 指定填充模式，填充值为0。它可以设置为： ``"zeros"`` 、 ``"reflect"`` 或 ``"replicate"`` 。默认值： ``"zeros"`` 。

        - **dtype** (:class:`mindspore.dtype`，可选) - Parameters的dtype。默认值： ``None``，使用 ``mstype.float32`` 。

    变量：
        - **weight** (Tensor) - 卷积层的权重，shape :math:`(C_{out}, C_{in} / \text{groups}, \text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})` 。
        - **bias** (Tensor) - 卷积层的偏置，shape :math:`(C_{out})` 。如果 `bias` 为 ``False``，则为 ``None``。

    输入：
        - **input** (Tensor) - shape为 :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` 或 :math:`(C_{in}, D_{in}, H_{in}, W_{in})` 的Tensor。

    输出：
        Tensor，shape为 :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` 或 :math:`(C_{out}, D_{out}, H_{out}, W_{out})`。

        padding为 ``"same"`` 时：

        .. math::
            \begin{array}{ll} \\
                D_{out} = \left \lceil{\frac{D_{in}}{\text{stride[0]}}} \right \rceil \\
                H_{out} = \left \lceil{\frac{H_{in}}{\text{stride[1]}}} \right \rceil \\
                W_{out} = \left \lceil{\frac{W_{in}}{\text{stride[2]}}} \right \rceil \\
            \end{array}

        padding为 ``"valid"`` 时：

        .. math::
            \begin{array}{ll} \\
                D_{out} = \left \lfloor{\frac{D_{in} - \text{dilation[0]} \times (\text{kernel_size[0]} - 1) - 1}
                {\text{stride[0]}}} \right \rfloor + 1 \\
                H_{out} = \left \lfloor{\frac{H_{in} - \text{dilation[1]} \times (\text{kernel_size[1]} - 1) - 1}
                {\text{stride[1]}}} \right \rfloor + 1 \\
                W_{out} = \left \lfloor{\frac{W_{in} - \text{dilation[2]} \times (\text{kernel_size[2]} - 1) - 1}
                {\text{stride[2]}}} \right \rfloor + 1 \\
            \end{array}

        padding为int或tuple/list时：

        .. math::
            \begin{array}{ll} \\
                D_{out} = \left \lfloor{\frac{D_{in} + 2 \times padding[0] - \text{dilation[0]} \times
                (\text{kernel_size[0]} - 1) - 1}{\text{stride[0]}} + 1} \right \rfloor \\
                H_{out} = \left \lfloor{\frac{H_{in} + 2 \times padding[1] - \text{dilation[1]} \times
                (\text{kernel_size[1]} - 1) - 1}{\text{stride[1]}} + 1} \right \rfloor \\
                W_{out} = \left \lfloor{\frac{W_{in} + 2 \times padding[2] - \text{dilation[2]} \times
                (\text{kernel_size[2]} - 1) - 1}{\text{stride[2]}} + 1} \right \rfloor \\
            \end{array}

    异常：
        - **ValueError** - `in_channels` 、 `out_channels` 、 `kernel_size` 、 `stride` 或 `dilation` 小于1。
        - **ValueError** - `padding` 小于0。
