mindspore.nn.AvgPool1d
=======================

.. py:class:: mindspore.nn.AvgPool1d(kernel_size=1, stride=1, pad_mode="valid", padding=0, ceil_mode=False, count_include_pad=True)

    在一个输入Tensor上应用1D平均池化运算，可被视为组成一个1D输入平面。

    通常，输入的shape为 :math:`(N_{in}, C_{in}, L_{in})` ，AvgPool1d在 :math:`(L_{in})` 维度上输出区域平均值。
    给定 `kernel_size` 为 :math:`l_{ker}` ， `stride` 为 :math:`s_0` ，公式定义如下：

    .. math::
        \text{output}(N_i, C_j, l) = \frac{1}{l_{ker}} \sum_{n=0}^{l_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times l + n)

    参数：
        - **kernel_size** (int) - 指定池化核尺寸大小，数据类型为整型。默认值： ``1`` 。
        - **stride** (int) - 池化操作的移动步长，数据类型为整型。默认值： ``1`` 。
        - **pad_mode** (str，可选) - 指定填充模式，填充值为0。可选值为 ``"same"`` ， ``"valid"`` 或 ``"pad"`` 。默认值： ``"valid"`` 。

          - ``"same"``：在输入的两端填充，使得当 `stride` 为 ``1`` 时，输入和输出的shape一致。待填充的量由算子内部计算，若为偶数，则均匀地填充在四周，若为奇数，多余的填充量将补充在右端。如果设置了此模式， `padding` 必须为0。
          - ``"valid"``：不对输入进行填充，返回输出可能的最大长度，不能构成一个完整stride的额外的像素将被丢弃。如果设置了此模式， `padding` 必须为0。
          - ``"pad"``：对输入填充指定的量。在这种模式下，填充的量由 `padding` 参数指定。如果设置此模式， `padding` 必须大于或等于0。

        - **padding** (Union(int, tuple[int], list[int])) - 池化填充值，只有 `pad` 模式才能设置为非0。默认值： ``0`` 。 `padding` 只能是一个整数或者包含一个整数的tuple/list，设定后，则会在输入的左边和右边填充 `padding` 次或者 `padding[0]` 次。
        - **ceil_mode** (bool) - 若为 ``True`` ，使用ceil来计算输出shape。若为 ``False`` ，使用floor来计算输出shape。默认值： ``False`` 。
        - **count_include_pad** (bool) - 如果为 ``True`` ，平均计算将包括零填充。默认值： ``True`` 。

    输入：
        - **x** (Tensor) - shape为 :math:`(N, C_{in}, L_{in})` 或 :math:`(C_{in}, L_{in})` 的Tensor。

        其中，如果 `pad_mode` 为 `pad` 模式时，输出的shape计算公式如下：

        .. math::
            L_{out} = \left\lfloor \frac{L_{in} +
            2 \times \text{padding} - \text{kernel_size}}{\text{stride}} + 1\right\rfloor

    输出：
        shape为 :math:`(N, C_{out}, L_{out})` 或 :math:`(C_{out}, L_{out})` 的Tensor。

    异常：
        - **TypeError** - `kernel_size` 或 `stride` 不是int。
        - **ValueError** - `pad_mode` 既不是"valid"，也不是"same" 或者 "pad"，不区分大小写。
        - **ValueError** - `kernel_size` 或 `stride` 小于1。
        - **ValueError** - `padding` 为tuple/list时长度不为1。
        - **ValueError** -  `x` 的shape长度不等于2或3。
        - **ValueError** -  `pad_mode` 不为 "pad" 的时候 `padding` 为非0。
