mindspore.ops.MaxUnpool2D
=========================

.. py:class:: mindspore.ops.MaxUnpool2D(ksize, strides=0, pads=0, output_shape=(), data_format="NCHW")

    MaxPool2D的逆过程。

    由于MaxPool2D会丢失非最大值，因此它不是完全可逆的。MaxUnpool2D将MaxPool2D的输出作为输入，包括最大值的索引，并计算部分逆，其中所有非最大值都被设置为零。
    例如，输入的shape为 :math:`(N, C, H_{in}, W_{in})` ，输出的shape为 :math:`(N, C, H_{out}, W_{out})` ，
    则该操作如下式所示：

    .. math::
        \begin{array}{ll} \\
        H_{out} = (H{in} - 1) \times strides[0] - 2 \times pads[0] + ksize[0] \\
        W_{out} = (W{in} - 1) \times strides[1] - 2 \times pads[1] + ksize[1] \\
        \end{array}

    参数：
        - **ksize** (Union[int, tuple[int]]) - 用于取最大值的内核大小，
          是一个整数，表示内核的高度和宽度，或一个元组包含两个整数，分别表示高度和宽度的整数。
        - **strides** (Union[int, tuple[int]]，可选) - 内核移动的步长，如果 `strides` 为0或(0, 0),
          那么 `strides` 等于 `ksize` 。默认值：0。

          - 一个整数，则表示移动的高度和宽度都是 `strides` ，
          - 两个整数的元组，分别表示高度和宽度方向的移动步长。
  
        - **pads** (Union[int, tuple[int]]，可选) - 需要被填充的值。默认值：0。

          - 如果 `pads` 是一个整数，则高度和宽度方向的填充数量相同，都等于 `pads` 。
          - 如果 `pads` 是含两个整数的元组，高度和宽度方向的填充数量分别等于 `pads[0]` 和 `pads[1]` 。
        
        - **output_shape** (tuple[int]，可选) - 一个可选的输入，指定目标输出的尺寸。默认值：()。
         
          - 如果 :math:`output\_shape == ()` ，则输出的shape由 `kszie` 、 `strides` 和 `pads` 计算得到。
          - 如果 :math:`output\_shape != ()` ，则 `output_shape` 必须为 :math:`(N, C, H, W)` 或 :math:`(N, H, W, C)` ，
            同时 `output_shape` 必须属于 :math:`[(N, C, H_{out} - strides[0], W_{out} - strides[1]),
            (N, C, H_{out} + strides[0], W_{out} + strides[1])]` 。
        
        - **data_format** (str，可选) - 可选的数据格式。当前支持 `NCHW` 和 `NHWC` 。默认值： `NCHW` 。

    输入：
        - **x** (Tensor) - 求逆的输入Tensor。其shape为 :math:`(N, C, H_{in}, W_{in})` 或 :math:`(N, H_{in}, W_{in}, C)` 。
        - **argmax** (Tensor) - `argmax` 表示最大值的索引。其shape必须与输入 `x` 相同。
          `argmax` 的值必须属于 :math:`[0, H_{in} \times W_{in} - 1]` ，数据类型必须为int32或int64。

    输出：
        Tensor，其shape为 :math:`(N, C, H_{out}, W_{out})` 或 :math:`(N, H_{out}, W_{out}, C)` ，
        与 `x` 的数据类型相同。

    异常：
        - **TypeError** - 如果 `x` 或 `argmax` 的数据类型不支持。
        - **TypeError** - 如果 `ksize` 、 `strides` 或 `pads` 既不是int又不是tuple。
        - **ValueError** - 如果 `strides` (也支持0和(0, 0)) 或 `ksize` 的值不是正数。
        - **ValueError** - 如果 `pads` 的值是负数。
        - **ValueError** - 如果  `ksize` 、 `strides` 或 `pads` 不是长度为2的tuple。
        - **ValueError** - 如果 `data_format` 不是str，同时也不是 `NCHW` 或 `NHWC` 。
        - **ValueError** - 如果 `output_shape` 的长度不是0或4。
        - **ValueError** - 如果 `output_shape` 不在由 `ksize` 、 `strides` 和 `pads` 计算出的输出尺寸范围内。
