mindspore.ops.AdaptiveMaxPool3D
===============================

.. py:class:: mindspore.ops.AdaptiveMaxPool3D

    对一个多平面输入信号执行三维自适应最大值池化。

    更多参考详见 :func:`mindspore.ops.adaptive_max_pool3d` 。

    输入：
        - **x** (Tensor) - shape为 :math:`(C, D, H, W)` 或 :math:`(N，C, D, H, W)` 的Tensor，支持的数据类型包括int8、int16、int32、int64、uint8、uint16、uint32、uint64、float16、float32、float64。
        - **output_size** (Union[int, tuple]) - 指定输出的size。可以用一个整数统一表示输出的深度、高度和宽度，或者用一个整数三元组来分别表示输出的深度、高度和宽度。指定的值必须是正整数。如果是None则表示对应维度输出和输入size相同。

    输出：
        - **y** (Tensor) - Tensor，与输入 `input` 的数据类型和维度相同。
        - **argmax** (Tensor) - Tensor，最大值对应的索引，数据类型为int32，并与 `y` 的shape相同。
