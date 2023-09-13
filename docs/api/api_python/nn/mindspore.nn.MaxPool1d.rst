mindspore.nn.MaxPool1d
=======================

.. py:class:: mindspore.nn.MaxPool1d(kernel_size=1, stride=1, pad_mode="valid", padding=0, dilation=1, return_indices=False, ceil_mode=False)

    在一个输入Tensor上应用1D最大池化运算，该Tensor可被视为一维平面的组合。

    通常，输入的shape为 :math:`(N_{in}, C_{in}, L_{in})` ，MaxPool1d输出 :math:`(L_{in})` 维度区域最大值。
    给定 `kernel_size` 为 :math:`ks = (l_{ker})` 和 `stride` 为 :math:`s = (s_0)`，公式如下：

    .. math::
        \text{output}(N_i, C_j, l) = \max_{n=0, \ldots, l_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times l + n)

    参数：
        - **kernel_size** (int) - 指定池化核尺寸大小。默认值： ``1`` 。
        - **stride** (int) - 池化操作的移动步长，数据类型为整型。默认值： ``1`` 。
        - **pad_mode** (str，可选) - 指定填充模式，填充值为0。可选值为 ``"same"`` ， ``"valid"`` 或 ``"pad"`` 。默认值： ``"valid"`` 。

          - ``"same"``：在输入的两端填充，使得当 `stride` 为 ``1`` 时，输入和输出的shape一致。待填充的量由算子内部计算，若为偶数，则均匀地填充在四周，若为奇数，多余的填充量将补充在右端。如果设置了此模式， `padding` 必须为0。
          - ``"valid"``：不对输入进行填充，返回输出可能的最大长度，不能构成一个完整stride的额外的像素将被丢弃。如果设置了此模式， `padding` 必须为0。
          - ``"pad"``：对输入填充指定的量。在这种模式下，填充的量由 `padding` 参数指定。如果设置此模式， `padding` 必须大于或等于0。

        - **padding** (Union(int, tuple[int], list[int])) - 池化填充值。默认值： ``0`` 。 `padding` 只能是一个整数或者包含一个整数的tuple/list，设定后，则会在输入的左边和右边填充 `padding` 次或者 `padding[0]` 次。
        - **dilation** (Union(int, tuple[int])) - 卷积核中各个元素之间的间隔大小，用于提升池化操作的感受野。如果为tuple，其长度只能为1。默认值： ``1`` 。
        - **return_indices** (bool) - 若为 ``True`` ，将会同时返回最大池化的结果和索引。默认值： ``False`` 。
        - **ceil_mode** (bool) - 若为 ``True`` ，使用ceil来计算输出shape。若为 ``False`` ，使用floor来计算输出shape。默认值： ``False`` 。

    输入：
        - **x** (Tensor) - shape为 :math:`(N, C_{in}, L_{in})` 或 :math:`(C_{in}, L_{in})` 的Tensor。

    输出：
        如果 `return_indices` 为False，则是shape为 :math:`(N, C_{out}, L_{out})` 或 :math:`(C_{out}, L_{out})` 的Tensor。数据类型与 `x` 一致。
        如果 `return_indices` 为True，则是一个包含了两个Tensor的Tuple，表示maxpool的计算结果以及生成max值的位置。

        - **output** (Tensor) - 最大池化结果，shape为 :math:`(N, C_{out}, L_{out})` 或 :math:`(C_{out}, L_{out})` 的Tensor。数据类型与 `x` 一致。
        - **argmax** (Tensor) - 最大值对应的索引。数据类型为int64。

        其中，如果 `pad_mode` 为 `pad` 模式时，输出的shape计算公式如下：

        .. math::
            L_{out} = \left\lfloor \frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                \times (\text{kernel_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    异常：
        - **TypeError** - `kernel_size` 或 `strides` 不是整数。
        - **ValueError** - `pad_mode` 既不是 ``"valid"`` ，也不是 ``"same"`` 或者 ``"pad"``，不区分大小写。
        - **ValueError** - `data_format` 既不是 ``'NCHW'`` 也不是 ``'NHWC'``。
        - **ValueError** - `kernel_size` 或 `strides` 小于1。
        - **ValueError** - `x` 的shape长度不等于2或3。
        - **ValueError** - 当 `pad_mode` 不为 ``"pad"`` 时，`padding`、 `dilation`、 `return_indices`、 `ceil_mode` 参数不为默认值。
        - **ValueError** - `padding` 参数为tuple/list时长度不为1。
        - **ValueError** - `dilation` 参数为tuple时长度不为1。
        - **ValueError** - `dilation` 参数不为int也不为tuple。
        - **ValueError** - `pad_mode` 不为 ``"pad"`` 的时候 `padding` 为非0。