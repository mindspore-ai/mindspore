mindspore.ops.AdaptiveAvgPool3D
================================

.. py:class:: mindspore.ops.AdaptiveAvgPool3D(output_size)

    三维自适应平均池化。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    更多参考详见 :func:`mindspore.ops.adaptive_avg_pool3d`。

    参数：
        - **output_size** (Union[int, tuple]) - 指定输出特征图的尺寸。可以为一个整数或者三元tuple。

    输入：
        - **x** (Tensor) - AdaptiveAvgPool3D的输入，为五维或四维的Tensor。

    输出：
        Tensor，数据类型与 `x` 相同。
