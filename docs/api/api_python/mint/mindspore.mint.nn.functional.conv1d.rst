mindspore.mint.nn.functional.conv1d
====================================

.. py:function:: mindspore.mint.nn.functional.conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor

    对输入Tensor计算三维卷积。通常，输入Tensor的shape为 :math:`(N, C_{in}, L_{in})` ，其中 :math:`N` 为batch size，:math:`C` 为通道数， :math:`L` 为特征序列长度。

    根据以下公式计算输出：

    .. math::

        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{ccor}({\text{weight}(C_{\text{out}_j}, k), \text{X}(N_i, k)})

    其中， :math:`bias` 为输出偏置，:math:`ccor` 为 `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_ 操作，
    :math:`weight` 为卷积核的值， :math:`X` 为输入的特征图。

    - :math:`i` 对应batch数，其范围为 :math:`[0, N-1]` ，其中 :math:`N` 为输入batch。

    - :math:`j` 对应输出通道，其范围为 :math:`[0, C_{out}-1]` ，其中 :math:`C_{out}` 为输出通道数，该值也等于卷积核的个数。

    - :math:`k` 对应输入通道数，其范围为 :math:`[0, C_{in}-1]` ，其中 :math:`C_{in}` 为输入通道数，该值也等于卷积核的通道数。

    因此，上面的公式中， :math:`{bias}(C_{\text{out}_j})` 为第 :math:`j` 个输出通道的偏置， :math:`{weight}(C_{\text{out}_j}, k)` 表示第 :math:`j` 个\
    卷积核在第 :math:`k` 个输入通道的卷积核切片， :math:`{X}(N_i, k)` 为特征图第 :math:`i` 个batch第 :math:`k` 个输入通道的切片。
    卷积核shape为 :math:`(\text{kernel_size},)` ，其中 :math:`\text{kernel_size}` 是卷积核的长度。若考虑到输入输出通道以及groups，则完整卷积核的shape为
    :math:`(C_{out}, C_{in} / \text{groups}, \text{kernel_size})` ，
    其中 `groups` 是分组卷积时在通道上分割输入 `x` 的组数。

    想更深入了解卷积层，请参考论文 `Gradient Based Learning Applied to Document Recognition <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_ 。

    参数：
        - **input** (Tensor) - shape为 :math:`(N, C_{in}, L_{in})` 或 :math:`(C_{in}, L_{in})` 的Tensor。
        - **weight** (Tensor) - shape为 :math:`(C_{out}, C_{in} / \text{groups}, \text{kernel_size})` ，则卷积核的大小为 :math:`(\text{kernel_size})` 。
        - **bias** (Tensor，可选) - 偏置Tensor，shape为 :math:`(C_{out})` 的Tensor。如果 `bias` 是 ``None`` ，将不会添加偏置。默认值： ``None`` 。
        - **stride** (Union[int, tuple[int], list[int]]，可选) - 一维卷积核的移动步长。数据类型为整型或者长度为1的整型tuple/list。默认值： ``1`` 。
        - **padding** (Union[int, tuple[int], list[int], str]，可选) - 输入的长度方向上填充的数量。数据类型为int或包含1个整数的tuple/list或string { ``"valid"`` ，  ``"same"`` } 。值应该要大于等于0。默认值： ``0`` 。

          - ``"same"``：在输入的四周填充，使得当 `stride` 为 ``1`` 时，输入和输出的shape一致。待填充的量由算子内部计算，若为偶数，则均匀地填充在两侧，若为奇数，多余的填充量将补充在右侧。若设置该模式， `stride` 的值必须为1。
          - ``"valid"``：不对输入进行填充，返回输出可能的最大长度，不能构成一个完整stride的额外的序列将被丢弃。

        - **dilation** (Union[int, tuple[int], list[int]]，可选) - 卷积核膨胀尺寸。可以为单个int，或者由1个int组成的tuple/list。
          假设 :math:`dilation=(d)`, 则卷积核在长度方向间隔 :math:`d-1` 个元素进行采样。默认值： ``1`` 。
        - **groups** (int，可选) - 将过滤器拆分为组， `in_channels` 和 `out_channels` 必须可被 `groups` 整除。如果组数等于 `in_channels` 和 `out_channels` ，这个一维卷积层也被称为一维深度卷积层。默认值： ``1`` 。
          需满足以下约束条件：

          - :math:`(C_{in} \text{ % } \text{groups} == 0)` 
          - :math:`(C_{out} \text{ % } \text{groups} == 0)`
          - :math:`(C_{out} >= \text{groups})`
          - :math:`(\text{weight[1]} = C_{in} / \text{groups})`

    返回：
        Tensor，卷积后的值。shape为 :math:`(N, C_{out}, L_{out})` 。
        要了解不同的填充模式如何影响输出shape，请参考 :class:`mindspore.mint.nn.Conv1d` 以获取更多详细信息。

    异常：
        - **RuntimeError** - Ascend上受不同型号NPU芯片上L1缓存大小限制，用例尺寸或Kernel Size过大。
        - **TypeError** - 如果 `in_channels` ， `out_channels` 或者 `groups` 不是整数。
        - **TypeError** - 如果 `kernel_size` ， `stride`， 或者 `dilation` 既不是int也不是tuple。
        - **ValueError** - 输入特征图的大小与参数应满足输出公式，以确保输出特征图大小为正，否则会报错。
        - **ValueError** - 如果 `in_channels` ， `out_channels`， `kernel_size` ， `stride` 或者 `dilation` 小于1。
        - **ValueError** - 如果 `padding` 小于0。
        - **ValueError** - 如果 `padding` 是 ``"same"`` ， `stride` 不等于1。
        - **ValueError** - 输入参数不满足卷积输出公式。
        - **ValueError** - `kernel_size` 不能超过输入特征图的大小。
        - **ValueError** - `padding` 值不能导致计算区域超出输入大小。
