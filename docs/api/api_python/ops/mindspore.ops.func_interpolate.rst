mindspore.ops.interpolate
=========================

.. py:function:: mindspore.ops.interpolate(x, size=None, scale_factor=None, mode="nearest", align_corners=None, recompute_scale_factor=None)

    按照给定的 `size` 或 `scale_factor` 根据 `mode` 设置的插值方式，对输入 `x` 调整大小。

    参数：
        - **x** (Tensor) - 被调整大小的Tensor。输入向量必须为3维，4维或5维，形状为 `(batch, channels, [optional depth], [optional height], width)` ，数据类型为float。
        - **size** (Union[int, tuple[int], list[int]], 可选) - 目标大小。如果 `size` 为tuple或list，那么其长度应该和 `x` 维度相同。 `size` 和 `scale_factor` 同时只能指定一个。默认值：None。
        - **scale_factor** (Union[float, tuple[float], list[float]]，可选) - 每个维度的缩放系数。 `scales` 中的数全是正数。 `size` 和 `scale_factor` 同时只能指定一个。默认值：None。
        - **mode** (str) - 采样算法。以下采样方式的一种，'nearest', 'linear' (仅三维)，'bilinear' (仅四维)，'bicubic' (仅四维)，'trilinear' (仅五维)，'area'，'nearest-exact'(三维和四维)。默认值：'nearest'。
        - **align_corners** (bool) - 如果为True，缩放比例系数使用 `(new\_height - 1) / (height - 1)` 计算，此种方式调整的数据与原始数据边角对齐。如果为False，缩放系数通过 `new\_height / height` 计算。

          .. code-block::

              old_i = new_length != 1 ? new_i * (old_length - 1) / (new_length - 1) : 0  # 'align_corners' 为 True
              old_i = new_length > 1 ? (new_x + 0.5) * old_length / new_length - 0.5 : 0  # 'align_corners' 为 False

          此选项只对'linear'、'bilinear'、'bicubic'和'trilinear'模式有效。默认值：False。
        - **recompute_scale_factor** (bool, 可选) - 重计算 `scale_factor` 。如果为True，会使用参数 `scale_factor` 计算参数 `size`，最终使用 `size` 的值进行缩放。如果为False，将使用 `size` 或 `scale_factor` 直接进行插值。默认值：None。

    参数支持列表和支持平台：

    +----------------+------+----------------+---------------+------------------+
    | mode           | dim  | align_corners  | scale_factor  | device           |
    +================+======+================+===============+==================+
    | nearest        | 3    | \-             | ×             | Ascend,GPU,CPU   |
    +----------------+------+----------------+---------------+------------------+
    |                | 4    | \-             | ×             | Ascend,GPU,CPU   |
    +----------------+------+----------------+---------------+------------------+
    |                | 5    | \-             | √             | GPU,CPU          |
    +----------------+------+----------------+---------------+------------------+
    | linear         | 3    | √              | ×             | GPU,CPU          |
    +----------------+------+----------------+---------------+------------------+
    | bilinear       | 4    | √              | ×             | Ascend,GPU,CPU   |
    +----------------+------+----------------+---------------+------------------+
    | trilinear      | 5    | √              | √             | GPU,CPU          |
    +----------------+------+----------------+---------------+------------------+
    | bicubic        | 4    | √              | ×             | GPU,CPU          |
    +----------------+------+----------------+---------------+------------------+
    | area           | 3    | \-             | √             | Ascend,GPU,CPU   |
    +----------------+------+----------------+---------------+------------------+
    |                | 4    | \-             | √             | GPU              |
    +----------------+------+----------------+---------------+------------------+
    |                | 5    | \-             | √             | GPU,CPU          |
    +----------------+------+----------------+---------------+------------------+
    | nearest-exact  | 3    | \-             | ×             | Ascend,CPU       |
    +----------------+------+----------------+---------------+------------------+
    |                | 4    | \-             | ×             | Ascend,CPU       |
    +----------------+------+----------------+---------------+------------------+

    - `-` 表示无此参数。
    - `×` 表示当前不支持此参数。
    - `√` 表示当前支持此参数。

    返回：
        调整大小之后的Tensor，维度和数据类型与 `x` 相同。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `x` 的数据类型不支持。
        - **TypeError** - `scales` 不是float类型的tuple。
        - **ValueError** - `scales` 中的数不全是正数。
        - **TypeError** - `sizes` 不是int64类型的tuple。
        - **ValueError** - `sizes` 中的数不全是正数。
        - **TypeError** - `coordinate_transformation_mode` 不是string。
        - **ValueError** - `coordinate_transformation_mode` 不在支持的列表中。
        - **TypeError** - `mode` 不是string类型。
        - **ValueError** - `mode` 不在支持的列表中。
