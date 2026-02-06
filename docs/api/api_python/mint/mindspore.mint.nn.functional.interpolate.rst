mindspore.mint.nn.functional.interpolate
========================================

.. py:function:: mindspore.mint.nn.functional.interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None, recompute_scale_factor=None)

    按照给定的 `size` 或 `scale_factor` 根据 `mode` 设置的插值方式，对输入 `input` 进行插值。

    .. note::
        - 在nearest模式下，在输入为3-D/4-D Tensor图像按 `scale_factor` 进行缩放的场景中可能存在精度问题.
        - `mode` 和 `recompute_scale_factor` 只能为常量。

    参数：
        - **input** (Tensor) - 被调整大小的Tensor。输入向量必须为三维，四维或五维，shape为 :math:`(N, C, [optional D], [optional H], W)` ，数据类型为float。
        - **size** (Union[int, tuple[int], list[int]], 可选) - 目标大小。如果 `size` 为tuple或list，那么其长度应该和 `input` 去掉 `N, C` 的维度相同。 `size` 和 `scale_factor` 同时只能指定一个。默认值： ``None`` 。
        - **scale_factor** (Union[float, tuple[float], list[float]]，可选) - 每个维度的缩放系数。如果 `scale_factor` 为tuple或list，那么其长度应该和 `input` 去掉 `N, C` 的维度相同。 `size` 和 `scale_factor` 同时只能指定一个。默认值： ``None`` 。
        - **mode** (str) - 采样算法。以下采样方式的一种，'nearest'(最近邻插值)， 'linear' (线性插值，仅三维)，'bilinear' (双线性插值，仅四维)，'trilinear'(三线性插值，仅五维)，'bicubic' (双三次插值，仅四维)。默认值： ``"nearest"`` 。
        - **align_corners** (bool, 可选) - 是否使用角对齐进行坐标映射。假设对输入Tensor沿x轴进行变换，具体计算公式如下：

          .. code-block::

              ori_i = new_length != 1 ? new_i * (ori_length - 1) / (new_length - 1) : 0   # 'align_corners' 为 True

              ori_i = new_length > 1 ? (new_i + 0.5) * ori_length / new_length - 0.5 : 0  # 'align_corners' 为 False

          其中， :math:`ori\_length` 与 :math:`new\_length` 分别表示Tensor在x轴方向上转换前、后的长度， :math:`new\_i` 表示转换后沿x轴第i个元素的坐标， :math:`ori\_i` 表示沿x轴原始数据的对应坐标。

          此选项只对 ``'linear'`` 、 ``'bilinear'`` 和 ``'bicubic'`` 模式有效，默认值： ``None``。

        - **recompute_scale_factor** (bool, 可选) - 重计算 `scale_factor` 。如果为True，会使用参数 `scale_factor` 计算参数 `size`，最终使用 `size` 的值进行缩放。如果为False，将使用 `size` 或 `scale_factor` 直接进行插值。默认值： ``None`` 。

    参数支持列表和支持平台：

    +---------------+-----------+---------------+--------------+----------------+
    | mode          | input.dim | align_corners | scale_factor | device         |
    +===============+===========+===============+==============+================+
    | nearest       | 3         | \-            | √            | Ascend         |
    +---------------+-----------+---------------+--------------+----------------+
    |               | 4         | \-            | √            | Ascend         |
    +---------------+-----------+---------------+--------------+----------------+
    |               | 5         | \-            | √            | Ascend         |
    +---------------+-----------+---------------+--------------+----------------+
    | linear        | 3         | √             | √            | Ascend         |
    +---------------+-----------+---------------+--------------+----------------+
    | bilinear      | 4         | √             | √            | Ascend         |
    +---------------+-----------+---------------+--------------+----------------+
    | bicubic       | 4         | √             | √            | Ascend         |
    +---------------+-----------+---------------+--------------+----------------+
    | trilinear     | 5         | √             | √            | Ascend         |
    +---------------+-----------+---------------+--------------+----------------+

    - `-` 表示无此参数。
    - `×` 表示当前不支持此参数。
    - `√` 表示当前支持此参数。

    返回：
        采样后的Tensor，维度和数据类型与 `input` 相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **ValueError** - `size` 和 `scale_factor` 都不为空。
        - **ValueError** - `size` 和 `scale_factor` 都为空。
        - **ValueError** - `size` 为元组或列表类型时长度不等于 `input.ndim - 2` 。
        - **ValueError** - `scale_factor` 为元组或列表类型时长度不等于 `input.ndim - 2` 。
        - **ValueError** - `mode` 不在模式支持列表中。
        - **ValueError** - `input.ndim` 不在模式对应维度的支持列表中。
        - **ValueError** - `size` 不为空， `recompute_scale_factor` 不为空。
        - **ValueError** - `scale_factor` 不在对应的支持列表中。
        - **ValueError** - `align_corners` 不在对应的支持列表中。
