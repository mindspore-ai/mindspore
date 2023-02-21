mindspore.nn.Upsample
=====================

.. py:class:: mindspore.nn.Upsample(size=None, scale_factor=None, mode="nearest", align_corners=None)

    根据指定的 `size` 或 `scale_factor` 对Tensor进行采样，过程使用上采样算法。

    必须指定 `size` 或 `scale_factor` 中的一个值，并且不能同时指定两者。

    参数：
        - **size** (Union(int, tuple[int], list[int])，可选) - 采样尺寸。可以是含有一个、两个或三个元素的list或者tuple。如果包含三个元素，则分别表示 :math:`([new\_depth], [new\_height], new\_width)` 。 `size` 和 `scale_factor` 有且只有一个可以被设为None。目前该参数仅支持设为None。默认值：None。
        - **scale_factor** (Union(float, tuple[float], list[float])，可选) - 采样Tensor的缩放因子，值为正整数。 `size` 和 `scale_factor` 有且只有一个可以被设为None。目前该参数仅支持设为None。默认值：None。
        - **mode** (str，可选) - 指定的上采样算法。可选值为：'nearest'、'linear' (仅支持3D输入)、 'bilinear'、'bicubic' (仅支持4D输入)、'trilinear' (仅支持5D输入)。默认值：'nearest'。
        - **align_corners** (bool，可选) - 如果为True，使用 :math:`(new\_height - 1) / (height - 1)` 对输入进行缩放使输入数据和缩放后数据的角落对齐。如果为False，则使用 :math:`new\_height / height` 进行缩放。默认值：None，此时不指定 `align_corners` ，赋值为False。

    输入：
        - **x** (Tensor) - 进行尺寸调整的Tensor。输入必须为3-D、4-D或5-D。其数据格式为： :math:`(batch, channels, depth(4-D或5-D才有此维度), height(5-D才有此维度), width)` 。数据类型为float16或float32。

    输出：
        Tensor。维度为3-D、4-D或5-D，数据类型与 `x` 相同。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `mode` 不是'nearest'、'linear'、'bilinear'、'bicubic'或'trilinear'。
        - **TypeError** - `size` 不是list、tuple或None。
        - **TypeError** - `scale_factor` 不为int或None。
        - **TypeError** - `align_corners` 不是bool。
        - **TypeError** - `x` 的数据类型不是float16或float32。
        - **ValueError** - `size` 和 `scale_factor` 同时为None或同时不为None。
        - **ValueError** - `x` 或 `size` 与 `mode` 不匹配。
        - **ValueError** -  `scale_factor` 小于零。
        - **ValueError** -  `size` 的长度和 `mode` 不匹配。
