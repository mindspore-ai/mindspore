mindspore.dataset.vision.c_transforms.RandomCropWithBBox
========================================================

.. py:class:: mindspore.dataset.vision.c_transforms.RandomCropWithBBox(size, padding=None, pad_if_needed=False, fill_value=0, padding_mode=Border.CONSTANT)

    对输入图像进行随机位置裁剪并相应地调整边界框。

    **参数：**

    - **size**  (Union[int, sequence]) - 裁剪图像的输出大小。大小值必须为正。
      如果 size 是整数，则返回大小为 (size, size) 的正方形裁剪。
      如果 size 是一个长度为 2 的序列，它应该是 (高度, 宽度)。
    - **padding**  (Union[int, sequence], 可选) - 填充图像的像素数。填充值必须非负值, 默认值：None。
      如果 `padding` 不为 None，则首先使用 `padding` 填充图像。
      如果 `padding` 是一个整数，代表为图像的所有方向填充该值大小的像素。
      如果 `padding` 是一个包含2个值的元组或列表，第一个值会用于填充图像的左侧和上侧，第二个值会用于填充图像的右侧和下侧。
      如果 `padding` 是一个包含4个值的元组或列表，则分别填充图像的左侧、上侧、右侧和下侧。
    - **pad_if_needed**  (bool, 可选) - 如果输入图像高度或者宽度小于 `size` 指定的输出图像大小，是否进行填充。默认值：False，不填充。
    - **fill_value**  (Union[int, tuple], 可选) - 边框的像素强度，仅对padding_mode Border.CONSTANT有效。
      如果是3元组，则分别用于填充R、G、B通道。
      如果是整数，则用于所有 RGB 通道。
      fill_value 值必须在 [0, 255] 范围内, 默认值：0。
    - **padding_mode**  (Border, 可选) - 边界填充方式, 默认值：Border.CONSTANT。它可以是 [Border.CONSTANT、Border.EDGE、Border.REFLECT、Border.SYMMETRIC] 中的任何一个。

      - **Border.CONSTANT** - 使用常量值进行填充。
      - **Border.EDGE** - 使用各边的边界像素值进行填充。
      - **Border.REFLECT** - 以各边的边界为轴进行镜像填充，忽略边界像素值。
      - **Border.SYMMETRIC** - 以各边的边界为轴进行对称填充，包括边界像素值。

    **异常：**

    - **TypeError** - 如果 `size` 不是int或sequence类型或元素不为int类型。
    - **TypeError** - 如果 `padding` 不是int或sequence类型或元素不为int类型。
    - **TypeError** - 如果 `pad_if_needed` 不是bool类型。
    - **TypeError** - 如果 `fill_value` 不是int或sequence类型或元素不为int类型。
    - **TypeError** - 如果 `padding_mode` 不是 :class:`mindspore.dataset.vision.Border` 的类型。
    - **ValueError** - 如果 `size` 不是正数。
    - **ValueError** - 如果 `padding` 为负数。
    - **ValueError** - 如果 `fill_value` 不在 [0, 255] 范围内。
    - **RuntimeError** - 如果输入图像的shape不是 <H, W> 或 <H, W, C>。
