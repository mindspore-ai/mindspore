mindspore.ops.FractionalMaxPool3DWithFixedKsize
===============================================

.. py:class:: mindspore.ops.FractionalMaxPool3DWithFixedKsize(ksize, output_shape, data_format="NCDHW")

    此运算对由多个输入平面组成的输入信号进行3D分数最大池化。最大池化操作通过由目标输出大小确定的随机步长在 kD x kH x kW 区域中进行。

    输出特征的数量等于输入平面的数量。

    详细内容请参考论文 `Fractional MaxPooling by Ben Graham <https://arxiv.org/abs/1412.6071>`_ 。

    输入和输出的数据格式可以是“NCDHW”和“NDHWC”，N是批量大小，C是通道数，D是特征深度，H是特征高度，W是特征宽度。

    参数：
        - **ksize** (Union[float, tuple]) - 目标ksize是  :math:`(D, H, W)`，其可以是一个元组，或者一个单独的 `K` 组成  :math:`(K, K, K)`，
          指明了输入Tensor的窗口大小 :math:`(D, H, W)`。

        - **output_shape** (Union[int, tuple]) - 目标输出shape为 :math:`(D, H, W)`，输出shape可以是一个元组，或者一个单独的 `H` 组成的 :math:`(H, H, H)`，
          指明了输出Tensor的大小 :math:`(D, H, W)` 。

        - **data_format** (str，可选) - 可选的数据格式值，当前支持 `NCDHW` 和 `NHDWC`，默认为 `NCDHW`。

    输入：
        - **x** (Tensor) - 输入Tensor，是一个4-D或者5-D的Tensor。数据类型为：float16、float32、double、int32、int64。
          支持的shape :math:`(N, C, D_{in}, H_{in}, W_{in})` 或者 :math:`(N, D_{in}, H_{in}, W_{in}, C)`。

        - **random_samples** (Tensor) - 随机步长，是一个3-D Tensor，数据类型为：float16、float32、double，值在(0, 1)之间。支持的shape为 :math:`(N, C, 3)`。

    输出：
        - **y** (Tensor) - 一个Tensor，与 `x` 具有相同的dtype，shape为 :math:`(N, C, D_{out}, H_{out}, W_{out})` 
          或者 :math:`(N, D_{out}, H_{out}, W_{out}, C)`。

        - **argmax** (Tensor) - 一个Tensor，输出的索引值。与 `y` 具有相同的shape，dype为int32或者int64。

    异常：
        - **TypeError** - 如果 `input_x` 不是一个4-D或者5-D的Tensor。
        - **TypeError** - 如果 `random_samples` 不是一个3-D的Tensor。
        - **TypeError** - 如果 `x` 的数据类型不是：float16、float32、double、int32、int64。
        - **TypeError** - 如果 `random_samples` 的数据类型不是：float16、float32、double。
        - **TypeError** - 如果 `argmax` 不是int32、int64。
        - **ValueError** - 如果 `output_shape` 不是一个元组，并且 `output_shape` 长度不是3。
        - **ValueError** - 如果 `ksize` 是一个元组，并且 `ksize` 长度不是3。
        - **ValueError** - 如果 `output_shape` 或者 `ksize` 中的数值不是正数。
        - **ValueError** - 如果 `data_format` 不是'NCDHW'，也不是'NDHWC'。
        - **ValueError** - 如果 `input_x` 和 `random_samples` 的第一维大小不相等。
        - **ValueError** - 如果 `input_x` and `random_samples` 的第二维大小不相等。
        - **ValueError** - 如果 `random_samples` 的第三维大小不是3。
