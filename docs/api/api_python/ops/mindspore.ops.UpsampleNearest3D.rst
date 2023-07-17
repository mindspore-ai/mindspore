mindspore.ops.UpsampleNearest3D
===============================

.. py:class:: mindspore.ops.UpsampleNearest3D()

    执行最近邻上采样操作。

    此运算符使用指定的 `output_size` 或 `scales` 缩放因子放大输入体积，过程使用最近邻算法。

    必须指定 `output_size` 或 `scales` 中的一个值，并且不能同时指定两者。

    输入：
        - **x** (Tensor) - Shape为 :math:`(N, C, D_{in}, H_{in}, W_{in})` 的五维Tensor。支持的数据类型：[uint8, float16, float32, float64]。
        - **output_size** (Union[tuple[int], list[int]]) - 指定输出体积大小的元组或int列表。默认值： ``None`` 。
        - **scales** (Union[tuple[float], list[float]]) - 指定上采样因子的元组或float列表。默认值： ``None`` 。

    输出：
        - **y** (Tensor) - 上采样输出。其shape :math:`(N, C, D_{out}, H_{out}, W_{out})` ，数据类型与 `x` 相同。

    异常：
        - **TypeError** - 当 `output_size` 不是 ``None`` 并且 `output_size` 不是list[int]或tuple[int]。
        - **TypeError** - 当 `scales` 不是 ``None`` 并且 `scales` 不是list[float]或tuple[float]。
        - **TypeError** - `x` 的数据类型不是uint8、float16、float32或float64。
        - **ValueError** - `output_size` 不为 ``None`` 时含有负数或0。
        - **ValueError** - `scales` 不为 ``None`` 时含有负数或0。
        - **ValueError** - `x` 维度不为5。
        - **ValueError** - `scales` 和 `output_size` 同时被指定或都不被指定。
        - **ValueError** - `scales` 被指定时其含有的元素个数不为3。
        - **ValueError** -  `output_size` 被指定时其含有的元素个数不为3。
