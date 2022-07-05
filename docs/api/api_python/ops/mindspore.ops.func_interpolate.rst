mindspore.ops.interpolate
=========================

.. py:function:: mindspore.ops.interpolate(x, roi=None, scales=None, sizes=None, coordinate_transformation_mode="align_corners", mode="linear")

    使用插值函数resize输入 `x` 。

    .. warning::
        - 实验特性，接口可能发生变化。
        - `roi` 是保留输入， `crop_and_resize` 坐标变换模式下生效，当前不支持。
        - Ascend平台下，当前不支持将 `mode` 设置为"linear"。
        - CPU平台下，当 `mode` 是"bilinear"时，当前不支持将 `coordinate_transformation_mode` 设置为"half_pixel"。

    **参数：**

    - **x** (Tensor) - 输入，3到5维的Tensor。
    - **roi** (tuple[float]， 可选) -  在 `crop_and_resize` 坐标变换模式下生效，当前不支持。
    - **scales** (tuple[float]， 可选) - 输入shape每个维度resize的系数。 `scales` 的长度跟 `x` 的shape长度相同。 `scales` 和 `size` 同时只能指定一个。
    - **sizes** (tuple[int]， 可选) - 输入shape指定轴的新维度。 `scales` 和 `size` 同时只能指定一个。当 `mode` 是"linear"时, `size` 为1个int元素 :math:`(new\_width,)` 的tuple。当 `mode` 是"bilinear"时, `size` 为2个int元素 :math:`(new\_height, new\_width)` 的tuple。
    - **coordinate_transformation_mode** (str) - 指定进行坐标变换的方式，默认值是"align_corners", 还可选"half_pixel"和"asymmetric"。
    - **mode** (str) - 所使用的插值方式。 目前支持"linear"和"bilinear"插值方式。默认值: "linear"。


    **返回：**

    Tensor，数据类型与 `x` 相同。

    **异常：**

    - **TypeError** - `x` 不是Tensor。
    - **TypeError** - `scales` 不是float类型的tuple。
    - **TypeError** - `size` 不是int64类型的tuple。
    - **TypeError** - `coordinate_transformation_mode` 不是string。
    - **TypeError** - `coordinate_transformation_mode` 不在支持的列表中。
    - **TypeError** - `mode` 不是string类型。
    - **TypeError** - `mode` 不在支持的列表中。