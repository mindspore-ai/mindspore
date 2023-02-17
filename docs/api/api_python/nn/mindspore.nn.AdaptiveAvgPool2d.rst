mindspore.nn.AdaptiveAvgPool2d
==============================

.. py:class:: mindspore.nn.AdaptiveAvgPool2d(output_size)

    对输入Tensor，提供二维的自适应平均池化操作。也就是说，对于输入任何尺寸，指定输出的尺寸都为H * W。但是输入和输出特征的数目不会变化。

    输入和输出数据格式可以是"NCHW"和"CHW"。N是批处理大小，C是通道数，H是特征高度，W是特征宽度。运算如下：

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
        - **output_size** (Union[int, tuple]) - 输出特征图的尺寸为H * W。可以是int类型的H和W组成的tuple，也可以为一个int值，代表相同H和W，或None，如果是None，则意味着输出大小与输入相同。

    输入：
        - **input** (Tensor) - AdaptiveAvgPool2d的输入，为三维或四维的Tensor，数据类型为float16、float32或者float64。

    输出：
        Tensor，输出shape为 :math:`(N, C_{out}, H_{out}, W_{out})`。

    异常：
        - **ValueError** - 如果 `output_size` 是tuple，并且 `output_size` 的长度不是2。
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - 如果 `input` 的数据类型不是float16、float32或者float64。
        - **ValueError** - 如果 `input` 的维度小于或等于 `output_size` 的维度。
