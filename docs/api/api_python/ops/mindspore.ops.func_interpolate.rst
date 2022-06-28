mindspore.ops.interpolate
=========================

.. py:function:: mindspore.ops.interpolate(x, roi=None, scales=None, sizes=None, coordinate_transformation_mode="align_corners", mode="linear")

    使用插值函数resize输入。

    .. warning::
        实验特性，接口可能发生变化。
        `roi` 是保留输入， `crop_and_resize` 坐标变换模式下生效，当前不支持。
    
    .. note::
        当前仅支持"linear"模式。

    **参数：**

    - **x** (Tensor) - 输入，3到5维的Tensor, 支持的数据类型有：uint8, int8, int16, int32, int64, float16, float, double。
    - **roi** (tuple[float]， 可选) -  在 `crop_and_resize` 坐标变换模式下生效，当前不支持。
    - **scales** (tuple[float]， 可选) - 输入shape每个维度resize的系数。 `scales` 和 `size` 同时只能指定一个。
    - **size** (tuple[float]， 可选) - 输入shape指定轴的新维度。 `scales` 和 `size` 同时只能指定一个。
    - **coordinate_transformation_mode** (str) - 指定进行坐标变换的方式，默认值是"align_corners", 还可选"half_pixel"和"asymmetric"。
    - **mode** (str) - 所使用的插值方式。 目前仅支持"linear"插值方式。


    **返回：**

    Tensor，shape与 `x` 相同， 数据类型是float32。

    **异常：**

    - **TypeError** - `x` 的DataType不在支持范围内。
    - **TypeError** - `scales` 的不是float类型的tuple。
    - **TypeError** - `size` 的不是int64类型的tuple。
    - **TypeError** - `coordinate_transformation_mode` 不是string。
    - **TypeError** - `coordinate_transformation_mode` 不在支持的列表中。
    - **TypeError** - `mode` 不是string。
    - **TypeError** - `mode` 不在支持的列表中。