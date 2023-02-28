mindspore.ops.adaptive_avg_pool2d
=================================

.. py:function:: mindspore.ops.adaptive_avg_pool2d(input, output_size)

    对一个多平面输入信号执行二维自适应最大值池化。也就是说，对于输入任何尺寸，指定输出的尺寸都为H * W。但是输入和输出特征的数目不会变化。

    输入和输出数据格式可以是"NCHW"和"CHW"。N是批处理大小，C是通道数，H是特征高度，W是特征宽度。

    对于二维的自适应平均池化操作，有如下公式：

    .. math::
        \begin{align}
        h_{start} &= floor(i * H_{in} / H_{out})\\
        h_{end} &= ceil((i + 1) * H_{in} / H_{out})\\
        w_{start} &= floor(j * W_{in} / W_{out})\\
        w_{end} &= ceil((j + 1) * W_{in} / W_{out})\\
        Output(i,j) &= \frac{\sum Input[h_{start}:h_{end}, w_{start}:w_{end}]}{(h_{end}- h_{start})
        * (w_{end}- w_{start})}
        \end{align}

    参数：
        - **input** (Tensor) - adaptive_avg_pool2d的输入，为三维或四维的Tensor，数据类型为float16、float32或者float64。
        - **output_size** (Union[int, tuple]) - 输出特征图的尺寸为H * W。可以是int类型的H和W组成的tuple，或代表相同H和W的一个int值，或None，如果是None，则意味着输出大小与输入相同。

    返回：
        Tensor，数据类型与 `input` 相同。

        输出的shape为 `input_shape[:len(input_shape) - len(out_shape)] + out_shape` 。

        .. math::
            out\_shape = \begin{cases}
            input\_x\_shape[-2] + output\_size[1], & \text{if output_size is (None, w);}\\
            output\_size[0] + input\_x\_shape[-1], & \text{if output_size is (h, None);}\\
            input\_x\_shape[-2:], & \text{if output_size is (None, None);}\\
            (h, h), & \text{if output_size is h;}\\
            (h, w), & \text{if output_size is (h, w)}
            \end{cases}

    异常：
        - **ValueError** - 如果 `output_size` 是tuple，并且 `output_size` 的长度不是2。
        - **ValueError** - 如果 `input` 的维度小于或等于 `output_size` 的维度。
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - 如果 `input` 的数据类型不是float16、float32或者float64。
