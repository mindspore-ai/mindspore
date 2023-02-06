mindspore.ops.fractional_max_pool3d
===================================

.. py:function:: mindspore.ops.fractional_max_pool3d(input_x, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)

    在输入 `input_x` 上应用三维分数最大池化。输出Tensor的shape可以由 `output_size` 和 `output_ratio` 其中之一确定，步长由 `_random_samples` 随机。 `output_size` 和 `output_ratio` 不能同时使用。

    分数最大池化的详细描述在 `Fractional MaxPooling by Ben Graham <https://arxiv.org/abs/1412.6071>`_ 。

    输入输出的数据格式可以是”NCDHW“。其中，N是批次大小，C是通道数，D是特征深度，H是特征高度，W是特征宽度。

    参数：
        - **input_x** (Tensor) - 四维或五维的Tensor，支持的数据类型：float16、float32、double、int32、int64。支持shape为 :math:`(N, C, D_{in}, H_{in}, W_{in})` 。
        - **kernel_size** (Union[int, tuple[int]]) - 指定池化核尺寸大小，如果为int，则代表池化核的深度，高和宽。如果为tuple，其值必须包含三个正int值分别表示池化核的深度，高和宽。取值必须为正int。
        - **output_size** (Union[int, tuple[int]]，可选) - 目标输出大小。如果是int，则表示输出目标的深、高和宽。如果是tuple，其值必须包含三个int值分别表示目标输出的深、高和宽。默认值：None。
        - **output_ratio** (Union[float, tuple[float]]，可选) - 目标输出shape与输入shape的比率。通过输入shape和 `output_ratio` 确定输出shape。支持数据类型：float16、float32、double，数值范围（0，1）。默认值：None。
        - **return_indices** (bool，可选) - 是否返回最大值的的索引值。默认值：False。
        - **_random_samples** (Tensor，可选) - 随机步长。支持的数据类型：float16、float32、double。shape为 :math:`(N, C, 3)` 的Tensor。数值范围（0，1）。默认值：None。

    返回：
        - **y** (Tensor) - 3D分数最大池化的输出，是一个Tensor。数据类型和输入相同，shape是 :math:`(N, C, D, H, W)` 。
        - **argmax** (Tensor) - 仅当 `return_indices` 为True时，输出最大池化的索引值。shape和输出 `y` 一致。

    异常：
        - **TypeError** - `input_x` 不是四维或五维Tensor。
        - **TypeError** - `random_samples` 不是三维Tensor。
        - **TypeError** - `x` 数据类型不是float16、float32、double、int32、int64。
        - **TypeError** - `random_samples` 数据类型不是float16、float32、double。
        - **TypeError** - `argmax` 数据类型不是int32、int64。
        - **ValueError** - `output_shape` 不是长度为3的tuple。
        - **ValueError** - `kernal_size` 不是长度为3的tuple。
        - **ValueError** - `output_shape` 和 `kernel_size` 不是正数。
        - **ValueError** - `output_size` 和 `output_ratio` 同时为 `None` 。
        - **ValueError** - `input_x` 和 `random_samples` 的第一维度大小不相等。
        - **ValueError** - `input_x` 和 `random_samples` 第二维度大小不相等。
        - **ValueError** - `random_samples` 第三维度大小不是3。
