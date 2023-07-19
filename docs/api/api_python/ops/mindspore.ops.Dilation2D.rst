mindspore.ops.Dilation2D
=========================

.. py:class:: mindspore.ops.Dilation2D(stride, dilation, pad_mode="SAME", data_format="NCHW")

    计算4-D和3-D输入Tensor的灰度膨胀。

    对输入的shape为 :math:`(N, C_{in}, H_{in}, W_{in})` ，应用2-D膨胀，其中，

    :math:`N` 为batch大小， :math:`H` 为高度， :math:`W` 为宽度， :math:`C` 为通道数量。

    给定kernel size :math:`ks = (h_{ker}, w_{ker})`, stride :math:`s = (s_0, s_1)`，和

    dilation :math:`d = (d_0, d_1)` ，计算如下：

    .. math::
        \text{output}(N_i, C_j, h, w) = \max_{m=0, \ldots, h_{ker}-1} \max_{n=0, \ldots, w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times h + d_0 \times m, s_1 \times w + d_1 \times n) + \text{filter}(C_j, m, n)

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    .. note::
        如果输入数据类型为float32，算子仍然按float16模式执行。

    参数：
        - **stride** (Union(int，tuple[int])) - kernel移动的距离。
          如果为一个int整数，则表示了height和width共同的步长。
          如果为两个int整数的元组，则分别表示height和width的步长。
          如果为四个int整数的元组，则说明数据格式为 ``NCHW`` ，表示 `[1, 1, stride_height, stride_width]` 。
        - **dilation** (Union(int，tuple[int])) - 数据类型为int，或者包含2个整数的元组，或者包含4个整数的元组，指定用于扩张卷积的膨胀速率。
          如果设置为 :math:`k > 1` ，则每次抽样点跳过 :math:`k - 1` 个像素点。
          其值必须大于等于1，并且以输入的宽度和高度为边界。
        - **pad_mode** (str，可选) - 指定填充模式，填充值为0。可选值为 ``"same"`` 或 ``"valid"``。默认值： ``"valid"`` 。

          - ``"same"``：在输入的四周填充，使得当 `stride` 为 ``1`` 时，输入和输出的shape一致。待填充的量由算子内部计算，若为偶数，则均匀地填充在四周，若为奇数，多余的填充量将补充在底部/右侧。
          - ``"valid"``：不对输入进行填充，返回输出可能的最大高度和宽度，不能构成一个完整stride的额外的像素将被丢弃。

        - **data_format** (str，可选) - 数据格式的值。目前只支持 ``NCHW`` ，默认值： ``NCHW`` 。

    输入：
        - **x** (Tensor) - 输入数据。一个四维Tensor，shape必须为
          :math:`(N, C_{in}, H_{in}, W_{in})` 。
        - **filter** (Tensor) - 一个三维Tensor，数据类型和输入 `x` 相同，shape必须为
          :math:`(C_{in}, H_{filter}, W_{filter})` 。

    输出：
        Tensor，其值已经过dilation2D。shape为 :math:`(N, C_{out}, H_{out}, W_{out})`，未必和输入 `x` shape相同，数据类型和输入 `x` 相同。

    异常：
        - **TypeError** - 如果输入 `x` 或者 `filter` 的数据类型不是uint8、uint16、uint32、uint64、int8、int16、
          int32、int64、float16、float32、float64。
        - **TypeError** - 如果参数 `stride` 或者 `dilation` 不是一个整数或者包含两个整数的元组或者包含四个整数的元组。
        - **ValueError** - 如果参数 `stride` 或者 `dilation` 是一个元组，并且它的长度不是2或者4。
        - **ValueError** - 如果参数 `stride` 或者 `dilation` 是一个包含四个整数的元组，它的shape不是 `(1, 1, height, width)`。
        - **ValueError** - 如果参数 `stride` 的取值范围不是 `[1, 255]` 。
        - **ValueError** - 如果参数 `dilation` 的值小于1。
        - **ValueError** - 如果参数 `pad_mode` 不是 `same` 、 `valid` 、 `SAME` 或者 `VALID`。
        - **ValueError** - 如果参数 `data_format` 不是字符串 `NCHW` 。
