mindspore.ops.AdaptiveMaxPool2D
===============================

.. py:class:: mindspore.ops.AdaptiveMaxPool2D(output_size)

    对一个多平面输入信号执行二维自适应最大值池化。

    更多参考详见 :func:`mindspore.ops.adaptive_max_pool2d`。

    参数：
        - **output_size** (Union[int, tuple]) - 输出特征图的尺寸为H * W。 `ouput_size` 可以是int类型的H和W组成的tuple。也可以为一个int值，代表相同H和W。H和W可以是None，则意味着输出大小与输入相同。

    输入：
        - **input_x** (Tensor) - AdaptiveMaxPool2d的输入，为三维或四维的Tensor，数据类型为float16、float32或者float64。

    输出：
        Tensor，数据类型与 `input_x` 相同。
