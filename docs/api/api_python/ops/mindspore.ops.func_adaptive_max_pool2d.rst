mindspore.ops.adaptive_max_pool2d
=================================

.. py:function:: mindspore.ops.adaptive_max_pool2d(input, output_size, return_indices=False)

    对输入Tensor，提供二维自适应最大池化操作。即对于输入任何尺寸，指定输出的尺寸都为H * W。但是输入和输出特征的数目不会变化。

    输入和输出数据格式可以是"NCHW"和"CHW"。N是批处理大小，C是通道数，H是特征高度，W是特征宽度。运算如下：

    .. math::

        \begin{align}
        h_{start} &= floor(i * H_{in} / H_{out})\\
        h_{end} &= ceil((i + 1) * H_{in} / H_{out})\\
        w_{start} &= floor(j * W_{in} / W_{out})\\
        w_{end} &= ceil((j + 1) * W_{in} / W_{out})\\
        Output(i,j) &= {\max Input[h_{start}:h_{end}, w_{start}:w_{end}]}
        \end{align}

    .. note::
        Ascend平台input参数仅支持float16类型。

    参数：
        - **input** (Tensor) - adaptive_max_pool2d的输入，为三维或四维的Tensor，数据类型为float16、float32或者float64。
        - **output_size** (Union[int, tuple]) - 输出特征图的尺寸为H * W。可以是int类型的H和W组成的tuple，或代表相同H和W的一个int值，或None，如果是None，则意味着输出大小与输入相同。
        - **return_indices** (bool) - 如果为True，输出最大值的索引，默认值为False。

    返回：
        Tensor，数据类型与 `input` 相同。

        输出的shape为 `input_shape[:len(input_shape) - len(out_shape)] + out_shape` 。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `input` 中的数据不是float16, float32, float64.
        - **TypeError** - `output_size` 不是int或者tuple。
        - **TypeError** - `return_indices` 不是bool。
        - **ValueError** - `output_size` 是tuple，但大小不是2。
        - **ValueError** - `input` 的维度不是CHW或者NCHW。
