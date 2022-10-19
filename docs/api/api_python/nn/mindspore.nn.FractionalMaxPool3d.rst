mindspore.nn.FractionalMaxPool3d
================================

.. py:class:: mindspore.nn.FractionalMaxPool3d(kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)

    对输入的多维数据进行三维的分数最大池化运算。

    对多个输入平面组成的输入上应用3D分数最大池化。在  :math:`(kD_{in}, kH_{in}, kW_{in})` 区域上应用最大池化操作，由输出shape决定随机步长。输出特征的数量等于输入平面的数量。

    分数最大池化的详细描述在 `Fractional MaxPooling by Ben Graham <https://arxiv.org/abs/1412.6071>`_ 。

    输入输出的数据格式可以是"NCDHW"。其中，"N"是批次大小，"C"是通道数，"D"是特征深度，"H"是特征高度，"W"是特征宽度。

    参数：
        - **kernel_size** (Union[float, tuple[int]]) - 指定池化核尺寸大小，如果为整数，则代表池化核的深、高和宽。如果为tuple，其值必须包含三个整数值分别表示池化核的深、高和宽。
        - **output_size** (Union[int, tuple[int]]) - 目标输出大小。如果是整数，则表示输出目标的深、高和宽。如果是tuple，其值必须包含三个整数值分别表示目标输出的深、高和宽。默认值是 `None` 。
        - **output_ratio** (Union[float, tuple[float]]) - 目标输出shape与输入shape的比率。通过输入shape和 `output_ratio` 确定输出shape。支持数据类型：float16、float32、double，数值介于0到1之间。默认值是 `None` 。
        - **return_indices** (bool) - 如果为 `True` ，返回分数最大池化的最大值的的索引值。默认值是 `False` 。
        - **_random_samples** (Tensor) - 随机步长。支持的数据类型：float16、float32、double。shape为 :math:`(N, C, 3)` 的Tensor。数值介于0到1之间。默认值是 `None` 。

    输入：
        - **input_x** (Tensor) - 4维或5维的张量，支持的数据类型：float16、float32、double、int32、int64。支持shape为 :math:`(N, C, D_{in}, H_{in}, W_{in})` 。

    输出：
        - **y** (Tensor) - 3D分数最大池化的输出，是一个张量。数据类型和输入相同，shape是 :math:`(N, C, output\underline{~}shape{D}, output\underline{~}shape{H}, output\underline{~}shape{W})` 。
        - **argmax** (Tensor) - 仅当 `return_indices` 为True时，输出最大池化的索引值。shape和输出 `y` 一致。

    异常：
        - **TypeError** - `input_x` 不是4维或5维张量。
        - **TypeError** - `random_samples` 不是3维张量。
        - **TypeError** - `x` 数据类型不是float16、float32、double、int32、int64。
        - **TypeError** - `random_samples` 数据类型不是float16、float32、double。
        - **TypeError** - `argmax` 数据类型不是int32、int64。
        - **ValueError** - `output_shape` 不是长度为3的元组。
        - **ValueError** - `kernal_size` 不是长度为3的元组。
        - **ValueError** - `output_shape` 和 `kernel_size` 不是正数。
        - **ValueError** - `output_size` 和 `output_ratio` 同时为 `None` 。
        - **ValueError** - `data_format` 数据格式不是 `NCDHW` 。
        - **ValueError** - `input_x` 和 `random_samples` 的第一维度大小不相等。
        - **ValueError** - `input_x` 和 `random_samples` 第二维度大小不相等。
        - **ValueError** - `random_samples` 第三维度大小不是3。
