mindspore.nn.FractionalMaxPool2d
================================

.. py:class:: mindspore.nn.FractionalMaxPool2d(kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)

    在输入上应用二维分数最大池化。输出Tensor的shape可以由 `output_size` 和 `output_ratio` 其中之一确定，步长由 `_random_samples` 随机决定。 `output_size` 和 `output_ratio` 不能同时使用且不能同时为None。

    分数最大池化的详细描述在 `Fractional Max-Pooling <https://arxiv.org/pdf/1412.6071>`_ 。

    参数：
        - **input** (Tensor) - shape为 :math:`(N, C, H_{in}, W_{in})` 的Tensor。支持的数据类型：float16、float32、float64、int32和int64。
        - **kernel_size** (Union[int, tuple[int]]) - 指定池化核尺寸大小，如果为正整数，代表池化核的高和宽。如果为tuple，其值必须为两个正整数，分别表示池化核的高和宽。
        - **output_size** (Union[int, tuple[int]]，可选) - 目标输出shape。如果为正整数，表示输出目标的高和宽。如果是tuple，其值必须包含两个正整数，分别表示目标输出的高和宽。为None时，输出shape由 `output_ration` 指定。默认值：None。
        - **output_ratio** (Union[float, tuple[float]]，可选) - 目标输出shape与输入shape的比率。通过输入shape和 `output_ratio` 确定输出shape。支持数据类型：float16、float32、float64，数值范围（0，1）。为None时，输出shape由 `output_size` 指定。默认值：None。
        - **return_indices** (bool，可选) - 是否返回最大值的的索引值。默认值：False。
        - **_random_samples** (Tensor，可选) - 分数最大池化的随机步长。数值范围为 :math:`(0, 1)` ，shape为 :math:`(N, C, 2)` 支持的数据类型：float16、float32、float64。为None时，不设置随机步长。默认值：None。

    返回：
        - **y** (Tensor) - 数据类型和输入相同，shape是 :math:`(N, C, H, W)`。
        - **argmax** (Tensor) - 输出的索引。shape和输出 `y` 一致，数据类型是int64。仅当 `return_indices` 为True时，才返回此输出。

    异常：
        - **TypeError** - `input` 不是float16、float32、float64、int32或int64。
        - **TypeError** - `_random_samples` 不是float16、float32或float64。
        - **ValueError** - `kernel_size` 不是int并且不是长度为2的tuple。
        - **ValueError** - `output_shape` 不是int并且不是长度为2的tuple。
        - **ValueError** - `kernel_size`， `output_shape` 与-1的和大于 `input` 的对应维度的量。
        - **ValueError** - `_random_samples` 维度不是3。
        - **ValueError** - `output_size` 和 `output_ratio` 同时为 `None` 。
        - **ValueError** - `input` 和 `_random_samples` 的第一维度大小不相等。
        - **ValueError** - `input` 和 `_random_samples` 第二维度大小不相等。
        - **ValueError** - `_random_samples` 第三维度大小不是2。
