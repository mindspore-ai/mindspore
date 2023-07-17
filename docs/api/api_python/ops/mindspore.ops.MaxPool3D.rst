mindspore.ops.MaxPool3D
========================

.. py:class:: mindspore.ops.MaxPool3D(kernel_size=1, strides=1, pad_mode="VALID", pad_list=0, ceil_mode=None, data_format="NCDHW")

    对输入的多维数据进行三维的最大池化运算。

    一般，输入shape为 :math:`(N_{in}, C_{in}, D_{in}, H_{in}, W_{in})` 的Tensor，输出 :math:`(D_{in}, H_{in}, W_{in})` 维上的区域最大值。给定 `kernel_size` 为 :math:`ks = (d_{ker}, h_{ker}, w_{ker})` 和 stride 为 :math:`s = (s_0, s_1, s_2)`，运算如下：

    .. math::
        \text{output}(N_i, C_j, d, h, w) =
        \max_{l=0, \ldots, d_{ker}-1} \max_{m=0, \ldots, h_{ker}-1} \max_{n=0, \ldots, w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times d + l, s_1 \times h + m, s_2 \times w + n)

    参数：
        - **kernel_size** (Union[int, tuple[int]]) - 指定池化核尺寸大小。整数类型，表示池化核深度、高和宽，或者是三个整数组成的元组，表示深、高和宽。默认值： ``1`` 。
        - **strides** (Union[int, tuple[int]]) - 池化操作的移动步长，整数类型，表示深、高和宽的移动步长，或者是三个整数组成的元组，表示深、高和宽移动步长。默认值： ``1`` 。
        - **pad_mode** (str，可选) - 指定填充模式，填充值为0。可选值为 ``"SAME"`` ， ``"VALID"`` 或 ``"PAD"`` 。默认值： ``"VALID"`` 。

          - ``"same"``：在输入的深度、高度和宽度维度进行填充，使得当 `stride` 为 ``1`` 时，输入和输出的shape一致。待填充的量由算子内部计算，若为偶数，则均匀地填充在四周，若为奇数，多余的填充量将补充在前方/底部/右侧。如果设置了此模式， `pad_list` 必须为0。
          - ``"valid"``：不对输入进行填充，返回输出可能的最大深度、高度和宽度，不能构成一个完整stride的额外的像素将被丢弃。如果设置了此模式， `pad_list` 必须为0。
          - ``"pad"``：对输入填充指定的量。在这种模式下，在输入的深度、高度和宽度方向上填充的量由 `pad_list` 参数指定。如果设置此模式， `pad_list` 必须大于或等于0。

        - **pad_list** (Union(int, tuple[int])) - 池化填充方式。默认值： ``0`` 。如果 `pad` 是一个整数，则头尾部、顶部，底部，左边和右边的填充都是相同的，等于 `pad` 。如果 `pad` 是六个整数的tuple，则头尾部、顶部、底部、左边和右边的填充分别等于填充pad[0]、pad[1]、pad[2]、pad[3]、pad[4]和pad[5]。  
        - **ceil_mode** (Union[bool, None]) - 是否使用ceil函数计算输出高度和宽度。默认值： ``None`` 。
        - **data_format** (str) - 输入和输出的数据格式。目前仅支持 ``'NCDHW'`` 。默认值： ``'NCDHW'`` 。

    输入：
        - **x** (Tensor) - shape为 :math:`(N, C, D_{in}, H_{in}, W_{in})` 的Tensor。数据类型为float16或float32。

    输出：
        Tensor，shape为 :math:`(N, C, D_{out}, H_{out}, W_{out})` 。数据类型与 `x` 相同。

    异常：
        - **TypeError** - `kernel_size` 或 `strides` 既不是int也不是元组。
        - **TypeError** - `pad_mode` 或 `data_format` 不是str。
        - **ValueError** - `kernel_size` 或 `strides` 不是正数。
        - **ValueError** - `pad_mode` 不是"SAME"，"VALID"，或"PAD"。
        - **ValueError** - `pad_mode` 取值为"SAME"或"VALID"， `ceil_mode` 取值不是None。
        - **ValueError** - `kernel_size` 或 `strides` 是长度不等于3的元组。
        - **ValueError** - `data_format` 不是'NCDHW'。