mindspore.ops.FractionalMaxPoolWithFixedKsize
=============================================

.. py:class:: mindspore.ops.FractionalMaxPoolWithFixedKsize(ksize, output_shape, data_format="NCHW")

    此运算对由多个输入平面组成的输入信号进行2D分数最大池化。最大池化操作通过由目标输出大小确定的随机步长在 `kH x kW` 区域中进行。

    对于任何输入大小，指定输出的大小为 `H x W` 。输出特征的数量等于输入平面的数量。

    详细内容请参考论文 `Fractional Max-Pooling <https://arxiv.org/pdf/1412.6071>`_ 。

    参数：
        - **ksize** (Union[int, tuple[int]]) - 用于取最大值的内核窗口的大小。目标 ksize 为 `H x W` 。ksize 可以是元组，也可以是 `K x K` 的单个K。
          指明了输入Tensor的窗口大小 `(H, W)` 。
        - **output_shape** (Union[int, tuple[int]]) - 目标输出shape为 `H x W` ，输出shape可以是一个元组，或者 `H x H` 的单个H。
          指明了输出tensor的大小 `(H, W)` 。
        - **data_format** (str，可选) - 可选的数据格式值，当前支持 `NCHW` ，默认为 `NCHW` 。

    输入：
        - **input_x** (Tensor) - Tensor的shape :math:`(N, C, H_{in}, W_{in})` ，数据类型为：float16、float32、float64、int32、int64。
        - **random_samples** (Tensor) - Tensor的shape为 :math:`(N, C, 2)` ，数据类型为：float16、float32、float64。

    输出：
        - **y** (Tensor) - 一个Tensor，与 `input_x` 具有相同的dtype，shape为 :math:`(N, C, output\underline{~}shape{H}, output\underline{~}shape{W})`。
        - **argmax** (Tensor) - 一个Tensor，数据类型必须为int64，与 `y` 具有相同的shape。

    异常：
        - **TypeError** - 如果 `input_x` 的数据类型不是：float16、float32、float64、int32、int64。
        - **TypeError** - 如果 `random_samples` 的数据类型不是：float16、float32、float64。
        - **ValueError** - 如果 `ksize` 不是一个数，并且不是一个长度为2的元组。 
        - **ValueError** - 如果 `output_shape` 不是一个数，并且不是长度为2的元组。
        - **ValueError** - 如果 `ksize` 、 `output_shape` 和-1的总和大于 `input_x` 的相应维度。
        - **ValueError** - 如果 `random_samples` 的轴数不是3。
        - **ValueError** - 如果 `input_x` 和 `random_samples` 的第一维大小不相等。
        - **ValueError** - 如果 `input_x` and `random_samples` 的第二维大小不相等。
        - **ValueError** - 如果 `random_samples` 的第三维大小不是2。
