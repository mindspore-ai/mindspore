mindspore.ops.ResizeLinear1D
============================

.. py:class:: mindspore.ops.ResizeLinear1D(coordinate_transformation_mode="align_corners")

    使用线性插值调整输入 `x` 为指定大小。

    调整输入 `x` 的宽。

    使用通用resize功能请参考 :func:`mindspore.ops.interpolate`。

    .. warning::
        - 实验特性，接口可能发生变化。
        - 目前，昇腾平台仅支持输入 `size` 为Tuple或List的场景。
        - 同时，昇腾平台上未支持属性coordinate_transformation_mode为asymmetric的情景。

    参数：
        - **coordinate_transformation_mode** (str) - 指定进行坐标变换的方式，默认值是"align_corners"，还可选"half_pixel"和"asymmetric"。

    输入：
        - **x** (Tensor) - ResizeBilinear的输入，三维的Tensor，其shape为 :math:`(batch, channels, width)`。支持以下数据类型：float16、float32、double。
        - **size** (Union[Tuple[int], List[int], Tensor[int]]) - 指定 `x` 宽的新尺寸，仅含一个整数 :math:`(new\_width)` 的Tuple、List或1-D Tensor。

    输出：
        Tensor，调整大小后的Tensor。shape为 :math:`(batch, channels, new\_width)` 的三维Tensor，数据类型和输入是一致的。

    异常：
        - **TypeError** - `x` 的数据类型不支持。
        - **TypeError** - `size` 不是Tuple[int]、List[int]或Tensor[int]。
        - **TypeError** - `coordinate_transformation_mode` 不是string。
        - **TypeError** - `coordinate_transformation_mode` 不在支持的列表中。
