mindspore.ops.interpolate
=========================

.. py:function:: mindspore.ops.interpolate(x, roi=None, scales=None, sizes=None, coordinate_transformation_mode="align_corners", mode="linear")

    使用 `mode` 设置的插值方式调整输入 `x` 大小。

    .. warning::
        - 实验特性，接口可能发生变化。
        - `roi` 是保留输入， `crop_and_resize` 坐标变换模式下生效，当前不支持。
        - Ascend平台下，当前不支持将 `mode` 设置为"linear"。
        - CPU平台下，当 `mode` 是"bilinear"时，当前不支持将 `coordinate_transformation_mode` 设置为"half_pixel"。

    参数：
        - **x** (Tensor) - 输入Tensor。当 `mode` 是"linear"时， `x` 为3维Tensor。当 `mode` 是"bilinear"时， `x` 为4维Tensor。
        - **roi** (tuple[float]，可选) - 在 `crop_and_resize` 坐标变换模式下生效，当前不支持。
        - **scales** (tuple[float]，可选) - 输入shape每个维度resize的系数。 `scales` 中的数全是正数。 `scales` 的长度跟 `x` 的shape长度相同。 `scales` 和 `sizes` 同时只能指定一个。
        - **sizes** (tuple[int]，可选) - 输入shape指定轴的新维度。 `sizes` 中的数全是正数。 `scales` 和 `sizes` 同时只能指定一个。当 `mode` 是"linear"时， `sizes` 为1个int元素 :math:`(new\_width,)` 的tuple。当 `mode` 是"bilinear"时， `sizes` 为2个int元素 :math:`(new\_height, new\_width)` 的tuple。
        - **coordinate_transformation_mode** (str) - 指定进行坐标变换的方式，默认值是"align_corners"，还可选"half_pixel"和"asymmetric"。
          假如我们需要将输入Tensor的x轴进行resize。我们记 `new_i` 为resize之后的Tenosr沿x轴的第i个坐标；记 `old_i` 为输入Tensor沿x轴的对应坐标；
          记 `new_length` 是resize之后的Tensor沿着x轴的长度，记 `old_length` 是输入Tensor沿x轴的长度。我们可以通过下面的公式计算出来 `old_i` ：

          .. code-block::

              old_i = new_length != 1 ? new_i * (old_length - 1) / (new_length - 1) : 0  # if set to 'align_corners'

              old_i = new_length > 1 ? (new_x + 0.5) * old_length / new_length - 0.5 : 0  # if set to 'half_pixel'

              old_i = new_length != 0 ? new_i * old_length / new_length : 0  # if set to 'asymmetric'

        - **mode** (str) - 所使用的插值方式。目前支持"linear"和"bilinear"插值方式。默认值："linear"。

    返回：
        Tensor，数据类型与 `x` 相同。

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