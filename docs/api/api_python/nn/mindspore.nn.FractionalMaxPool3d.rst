mindspore.nn.FractionalMaxPool3d
================================

.. py:class:: mindspore.nn.FractionalMaxPool3d(kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)

    在输入上应用三维分数最大池化。输出Tensor的shape可以由 `output_size` 和 `output_ratio` 其中之一确定，步长由 `_random_samples` 随机决定。 `output_size` 和 `output_ratio` 同时设置， `output_size` 会生效。 `output_size` 和 `output_ratio` 不能同时为 ``None`` 。

    分数最大池化的详细描述在 `Fractional MaxPooling by Ben Graham <https://arxiv.org/abs/1412.6071>`_ 。

    输入输出的数据格式可以是”NCDHW“。其中，N是批次大小，C是通道数，D是特征深度，H是特征高度，W是特征宽度。

    参数：
        - **kernel_size** (Union[int, tuple[int]]) - 指定池化核尺寸大小，如果为正整数，则代表池化核的深度，高和宽。如果为tuple，其值必须包含三个正整数值分别表示池化核的深度，高和宽。取值必须为正整数。
        - **output_size** (Union[int, tuple[int]]，可选) - 目标输出大小。如果是正整数，则表示输出目标的深、高和宽。如果是tuple，其值必须包含三个正整数值分别表示目标输出的深、高和宽。为 ``None`` 时，输出大小由 `output_ratio` 决定。默认值： ``None`` 。
        - **output_ratio** (Union[float, tuple[float]]，可选) - 目标输出shape与输入shape的比率。通过输入shape和 `output_ratio` 确定输出shape。支持数据类型：float16、float32、float64，数值介于0到1之间。为None时，输出大小由 `output_size` 决定。默认值： ``None`` 。
        - **return_indices** (bool，可选) - 是否返回最大值的索引值。默认值： ``False`` 。
        - **_random_samples** (Tensor，可选) - 随机步长。支持的数据类型：float16、float32、double。shape为 :math:`(N, C, 3)` 或 :math:`(1, C, 3)` 的Tensor。数值范围[0, 1)。默认值： ``None`` ， `_random_samples` 的值由区间[0, 1)上的均匀分布随机生成。

    输入：
        - **input** (Tensor) - 四维或五维的Tensor，支持的数据类型：float16、float32、float64。支持shape为 :math:`(N, C, D_{in}, H_{in}, W_{in})` 或 :math:`(C, D_{in}, H_{in}, W_{in})` 。

    输出：
        - **y** (Tensor) - 3D分数最大池化的输出，是一个Tensor。数据类型和 `input` 相同，shape是 :math:`(N, C, D_{out}, H_{out}, W_{out})` 或 :math:`(C, D_{out}, H_{out}, W_{out})` 。其中，:math:`(D_{out}, H_{out}, W_{out})` = `output_size` 或 :math:`(D_{out}, H_{out}, W_{out})` = `output_ratio` * :math:`(D_{in}, H_{in}, W_{in})` 。
        - **argmax** (Tensor) - 仅当 `return_indices` 为True时，输出最大池化的索引值。shape和输出 `y` 一致。

    异常：
        - **TypeError** - `input` 不是四维或五维Tensor。
        - **TypeError** - `_random_samples` 不是三维Tensor。
        - **TypeError** - `input` 数据类型不是float16、float32、double。
        - **TypeError** - `_random_samples` 数据类型不是float16、float32、double。
        - **TypeError** - `argmax` 数据类型不是int32、int64。
        - **TypeError** - `_random_samples` 和 `input` 数据类型不相同。
        - **ValueError** - `output_size` 不是长度为3的元组。
        - **ValueError** - `kernal_size` 不是长度为3的元组。
        - **ValueError** - `output_size` 和 `kernel_size` 不是正数。
        - **ValueError** - `output_size` 和 `output_ratio` 同时为 `None` 。
        - **ValueError** - `input` 和 `_random_samples` 的第一维度大小不相等。
        - **ValueError** - `input` 和 `_random_samples` 第二维度大小不相等。
        - **ValueError** - `_random_samples` 第三维度大小不是3。
