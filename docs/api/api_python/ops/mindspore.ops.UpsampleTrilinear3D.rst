mindspore.ops.UpsampleTrilinear3D
=================================

.. py:class:: mindspore.ops.UpsampleTrilinear3D(output_size=None, scales=None, align_corners=False)

    对于5D输入，在3D上用三线性插值进行上采样。

    使用指定的 `output_size` 或 `scales` 因子对容量输入进行缩放。其中 `output_size` 或 `scales` 必须给出一个，且不能同时指定。

    参数：
        - **output_size** (Union[tuple[int], list[int]]) - 指定输出卷大小的int型列表。默认值为None。 
        - **scales** (Union[tuple[float], list[float]]) - 指定上采样因子的浮点数列表。默认值为None。 
        - **align_corners** (bool) - True表示使用 :math:`(new\_height - 1) / (height - 1)` 进行缩放，使其输入图像与被调整之后的图像四角对齐；False表示使用 :math:`new\_height / height` 进行缩放。 默认值为False。

    输入：
        - **x** (Tensor) - shape: :math:`(N, C, D_{in}, H_{in}, W_{in})`。

    输出：
        Tensor，shape: :math:`(N, C, D_{out}, H_{out}, W_{out})`，数据类型与输入 `x` 相同。

    异常：
        - **TypeError** - `x` 的维度不为5。
        - **TypeError** - `x` 的数据类型不为float16，float32。
        - **TypeError** - `output_size` 的数据类型不为int型列表。
        - **TypeError** - `scales` 的数据类型不为float型列表。
        - **TypeError** - `align_corners` 不是一个布尔值。
        - **ValueError** - `output_size` 的类型为列表，其长度不为3。
        - **ValueError** - `scales` 的类型为列表，其长度不为3。
        - **ValueError** - `output_size` 和 `scales` 两者都为None。
        - **ValueError** - `output_size` 和 `scales` 两者都为非空列表。
