mindspore.ops.AdaptiveMaxPool2D
===============================

.. py:class:: mindspore.ops.AdaptiveMaxPool2D(output_size)

    对一个多平面输入信号执行二维自适应最大值池化。

    更多参考详见 :func:`mindspore.ops.adaptive_max_pool2d`。

    参数：
        - **output_size** (Union[int, tuple]) - 输出特征图的size。 `ouput_size` 可以为二元tuple表示 :math:`(H, W)`。或者是单个int表示 :math:`(H, H)` 。H、W可以是int或None，如果是None，则意味着输出的size与输入相同。

    输入：
        - **input_x** (Tensor) - `AdaptiveMaxPool2d` 的输入，为三维或四维的Tensor，数据类型为float16、float32或者float64。

    输出：
        Tensor，数据类型与 `input_x` 相同。
