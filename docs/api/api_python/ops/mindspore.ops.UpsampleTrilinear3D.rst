mindspore.ops.UpsampleTrilinear3D
=================================

.. py:class:: mindspore.ops.UpsampleTrilinear3D(align_corners=False)

    输入为五维度Tensor，跨其中三维执行三线性插值上调采样。

    此运算符使用指定的 `output_size` 或 `scales` 缩放因子放大输入体积，过程使用三线性上调算法。

    .. note::
        必须指定 `output_size` 或 `scales` 中的一个值，并且不能同时指定两者。

    参数：
        - **align_corners** (bool，可选) - 如果为 ``True`` ，则输入和输出Tensor由其角像素的中心点对齐，保留角像素处的值。如果为 ``False`` ，则输入和输出Tensor由其角像素的角点对齐，插值对边界外值使用边值填充。默认值： ``False``。

    输入：
        - **x** (Tensor) - Shape为 :math:`(N, C, D_{in}, H_{in}, W_{in})` 的五维Tensor。支持的数据类型：[float16, float32, float64]。
        - **output_size** (Union[tuple[int], list[int]]) - 包含3个int的元组或列表。元素分别为 :math:`(output\_depth, output\_height, output\_width)` 。默认值： ``None``。
        - **scales** (Union[tuple[float], list[float]]) - 包含3个float的元组或列表。元素分别为 :math:`(scale\_depth, scale\_height, scale\_width)` 。 默认值： ``None``。

    输出：
        - **y** (Tensor) - 上采样输出。其shape为 :math:`(N, C, D_{out}, H_{out}, W_{out})` ，数据类型与 `x` 相同。

    异常：
        - **TypeError** - 当 `output_size` 不是 ``None`` 并且 `output_size` 不是list[int]或tuple[int]。
        - **TypeError** - 当 `scales` 不是 ``None`` 并且 `scales` 不是list[float]或tuple[float]。
        - **TypeError** - `x` 的数据类型不是float16、float32或float64。
        - **TypeError** - `align_corners` 的数据类型不是bool。
        - **ValueError** - `output_size` 不为 ``None`` 时含有负数或0。
        - **ValueError** - `scales` 不为 ``None`` 时含有负数或0。
        - **ValueError** - `x` 维度不为5。
        - **ValueError** - `scales` 和 `output_size` 同时被指定或都不被指定。
        - **ValueError** - `scales` 被指定时其含有的元素个数不为3。
        - **ValueError** -  `output_size` 被指定时其含有的元素个数不为3。
