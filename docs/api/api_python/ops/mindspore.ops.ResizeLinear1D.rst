mindspore.ops.ResizeBilinear1D
==============================

.. py:class:: mindspore.ops.ResizeBilinear1D(coordinate_transformation_mode="align_corners")

    使用线性插值调整Tensor到指定的大小。

    调整输入三维Tensor的宽，且可以输入不同数据类型的图像，但输出的数据类型只能是float32。

    使用通用resize功能请参考 :func:`mindspore.ops.interpolate`。

    .. warning:::
        实验特性，接口可能发生变化。

    **参数：**

    - **coordinate_transformation_mode** (str) - 指定进行坐标变换的方式，默认值是"align_corners", 还可选"half_pixel"和"asymmetric"。

    **输入：**

    - **x** (Tensor) - ResizeBilinear的输入，三维的Tensor，其shape为 :math:`(batch, channels, width)`。支持以下数据类型：uint8, int8, int16, int32, int64, float16, float32, double。
    - **size** (Tensor) - 指定Tensor的新尺寸，一维的Tensor，其shape为 :math:`(1)` ，数据类型为int64。

    **输出：**

    Tensor，调整大小后的图像。shape为 :math:`(batch, channels,new\_width)` 的三维Tensor，数据类型是float32。 

    **异常：**

    - **TypeError** - `x` 的数据类型不支持。
    - **TypeError** - `size` 不是int64的数据类型。
    - **TypeError** - `coordinate_transformation_mode` 不是string。
    - **TypeError** - `coordinate_transformation_mode` 不在支持的列表中。
