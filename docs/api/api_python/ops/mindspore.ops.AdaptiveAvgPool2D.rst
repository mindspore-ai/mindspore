mindspore.ops.AdaptiveAvgPool2D
================================

.. py:class:: mindspore.ops.AdaptiveAvgPool2D(output_size)

    二维自适应平均池化。

    更多参考详见 :func:`mindspore.ops.adaptive_avg_pool2d`。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **output_size** (Union[int, tuple]) - 输出特征图的size。 `output_size` 可以为二元tuple表示 :math:`(H, W)`。或者是单个int表示 :math:`(H, H)` 。 :math:`H` 和 :math:`W` 可以是int或None，如果是None，则意味着输出的size与输入相同。

    输入：
        - **input_x** (Tensor) - AdaptiveAvgPool2D的输入，为三维或四维的Tensor，数据类型为float16、float32或者float64。

    输出：
        Tensor，数据类型与 `input_x` 相同。
