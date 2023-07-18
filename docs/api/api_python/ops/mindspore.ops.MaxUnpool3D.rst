mindspore.ops.MaxUnpool3D
=========================

.. py:class:: mindspore.ops.MaxUnpool3D(ksize, strides=0, pads=0, output_shape=(), data_format="NCDHW")

    :class:`mindspore.ops.MaxPool3D` 的逆过程。

    `MaxUnpool3D` 在计算逆的过程中，保留最大值位置的元素，并将非最大值位置元素设置为0。
    通常情况下，输入数据格式为 :math:`(N, C, D_{in}, H_{in}, W_{in})` ，
    输出数据的格式为 :math:`(N, C, D_{out}, H_{out}, W_{out})` ，计算公式如下：

    .. math::
        \begin{array}{ll} \\
        D_{out} = (D{in} - 1) \times strides[0] - 2 \times pads[0] + ksize[0] \\
        H_{out} = (H{in} - 1) \times strides[1] - 2 \times pads[1] + ksize[1] \\
        W_{out} = (W{in} - 1) \times strides[2] - 2 \times pads[2] + ksize[2] \\
        \end{array}

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **ksize** (Union[int, tuple[int]]) - 用于取最大值的内核大小，
          是一个整数，表示内核的深度、高度和宽度，或一个元组包含三个整数，分别表示深度、高度和宽度。
        - **strides** (Union[int, tuple[int]]，可选) - 内核移动的步长，默认值： ``0`` 。

          - 一个整数，则表示移动的深度、高度和宽度都是 `strides` ，
          - 三个整数的元组，分别表示深度、高度和宽度方向的移动步长。
          - 如果 `strides` 为0或(0, 0, 0)，那么 `strides` 等于 `ksize` 。
  
        - **pads** (Union[int, tuple[int]]，可选) - 指定填充量。默认值： ``0`` 。

          - 如果 `pads` 是一个整数，则深度、高度和宽度方向的填充数量相同，都等于 `pads` 。
          - 如果 `pads` 是含三个整数的元组，则深度、高度和宽度方向的填充数量分别等于 `pads[0]` 、 `pads[1]` 和 `pads[2]`。

        - **output_shape** (tuple[int]，可选) - 指定目标输出的尺寸。默认值： ``()`` 。

          - 如果 :math:`output\_shape == ()` ，则输出的shape由 `kszie` 、 `strides` 和 `pads` 根据上面的公式计算得到。
          - 如果 :math:`output\_shape != ()` ，则 `output_shape` 数据格式为 :math:`(N, C, D, H, W)` 或 :math:`(N, D, H, W, C)` ，
            同时 `output_shape` 必须属于 :math:`[(N, C, D_{out} - strides[0], H_{out} - strides[1], W_{out} - strides[2]),
            (N, C, D_{out} + strides[0], H_{out} + strides[1], W_{out} + strides[2])]` 范围。
        
        - **data_format** (str，可选) - 可选的数据格式。当前支持 ``'NCDHW'`` 和 ``'NDHWC'`` 。默认值： ``'NCDHW'`` 。

    输入：
        - **x** (Tensor) - 求逆的输入Tensor。其shape为 :math:`(N, C, D_{in}, H_{in}, W_{in})` 或 :math:`(N, D_{in}, H_{in}, W_{in}, C)` 。
        - **argmax** (Tensor) - `argmax` 表示最大值的索引。其shape必须与输入 `x` 相同。
          `argmax` 的值必须在 :math:`[0, D_{in} \times H_{in} \times W_{in} - 1]` 范围内，数据类型必须为int32或int64。

    输出：
        Tensor，其shape为 :math:`(N, C, D_{out}, H_{out}, W_{out})` 或 :math:`(N, D_{out}, H_{out}, W_{out}, C)` ，
        与 `x` 的数据类型相同。

    异常：
        - **TypeError** - 如果 `x` 或 `argmax` 的数据类型不是Number。
        - **TypeError** - 如果 `ksize` 、 `strides` 或 `pads` 既不是int又不是tuple。
        - **ValueError** - 如果 `strides` 或 `ksize` 的值是负数。
        - **ValueError** - 如果 `pads` 的值是负数。
        - **ValueError** - 如果  `ksize` 、 `strides` 或 `pads` 是tuple，但长度不是3。
        - **ValueError** - 如果 `data_format` 不是值为 "NCDHW" 或 "NDHWC" 的str。
        - **ValueError** - 如果 `output_shape` 的长度不是0或5。
        - **ValueError** - 如果 `output_shape` 不在由 `ksize` 、 `strides` 和 `pads` 计算出的输出尺寸范围内。

