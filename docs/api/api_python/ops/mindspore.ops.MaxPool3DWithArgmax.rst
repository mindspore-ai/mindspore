mindspore.ops.MaxPool3DWithArgmax
=================================

.. py:class:: mindspore.ops.MaxPool3DWithArgmax(ksize, strides, pads, dilation=(1, 1, 1), ceil_mode=False, data_format="NCDHW", argmax_type=mstype.int64)

    三维最大值池化，返回最大值结果及其索引值。

    输入是shape为 :math:`(N_{in}, C_{in}, D_{in}, H_{in}, W_{in})` 的Tensor，输出 :math:`(D_{in}, H_{in}, W_{in})` 维度中的最大值。给定 `ksize`
    :math:`ks = (d_{ker}, h_{ker}, w_{ker})`，和 `strides` :math:`s = (s_0, s_1, s_2)`，运算如下：

    .. math::
        \text{output}(N_i, C_j, d, h, w) =
        \max_{l=0, \ldots, d_{ker}-1} \max_{m=0, \ldots, h_{ker}-1} \max_{n=0, \ldots, w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times d + l, s_1 \times h + m, s_2 \times w + n)

    参数：
        - **ksize** (Union[int, tuple[int]]) - 池化核尺寸大小。可以是一个整数表示池化核的深度，高度和宽度，或者包含三个整数的tuple，分别表示池化核的深度，高度和宽度。
        - **strides** (Union[int, tuple[int]]) - 池化操作的移动步长。可以是一个整数表示在深度，高度和宽度方向的移动步长，或者包含三个整数的tuple，分别表示在深度，高度和宽度方向的移动步长。
        - **pads** (Union[int, tuple[int]]) - 池化填充长度。可以是一个整数表示在深度，高度和宽度方向的填充长度，或者包含三个整数的tuple，分别表示在深度，高度和宽度方向的填充长度。
        - **dilation** (Union[int, tuple[int]]) - 控制池化核内元素的间距。默认为(1, 1, 1)。
        - **ceil_mode** (bool) - 是否是用ceil代替floor来计算输出的shape。默认为False。
        - **data_format** (str) - 选择输入数据格式，当前仅支持'NDCHW'。默认为'NDCHW'。
        - **argmax_type** (mindspore.dtype) - 返回的最大值索引的数据类型。默认为mindspore.int64。

    输入：
        - **x** (Tensor) - shape为 :math:`(N_{in}, C_{in}, D_{in}, H_{in}, W_{in})` 的Tensor。支持数据类型包括int8、int16、int32、int64、uint8、uint16、uint32、uint64、float16、float32和float64。

    输出：
        包含两个Tensor的tuple，分别表示最大值结果和最大值对应的索引。

        - **output** (Tensor) - 输出的池化后的最大值，shape为 :math:`(N_{out}, C_{out}, D_{out}, H_{out}, W_{out})` 。其数据类型与 `x` 相同。
        - **argmax** (Tensor) - 输出的最大值对应的索引，数据类型为int32或者int64。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **ValueError** - `x` 的维度不是5D。
        - **TypeError** - `ksize` 、 `strides` 、`pads` 、`dilation` 不是int或者tuple。
        - **ValueError** - `ksize` 或 `strides` 的元素值小于1。
        - **ValueError** - `pads` 的元素值小于0。
        - **ValueError** - `data_format` 不是'NCDHW'。
        - **ValueError** - `argmax_type` 不是mindspore.int64或mindspore.int32。
