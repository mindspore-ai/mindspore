mindspore.dataset.vision.Pad
============================

.. py:class:: mindspore.dataset.vision.Pad(padding, fill_value=0, padding_mode=Border.CONSTANT)

    填充图像。

    参数：
        - **padding** (Union[int, Sequence[int, int], Sequence[int, int, int, int]]) - 图像各边填充的像素数。
          如果 `padding` 是一个整数，代表为图像的所有方向填充该值大小的像素。
          如果 `padding` 是一个包含2个值的元组或列表，第一个值会用于填充图像的左侧和右侧，第二个值会用于填充图像的上侧和下侧。
          如果 `padding` 是一个包含4个值的元组或列表，则分别填充图像的左侧、上侧、右侧和下侧。
          填充值必须为非负值。
        - **fill_value** (Union[int, tuple[int]], 可选) - 填充的像素值，仅在 `padding_mode` 取值为Border.CONSTANT时有效。
          如果是3元素元组，则分别用于填充R、G、B通道。
          如果是整数，则用于所有 RGB 通道。
          `fill_value` 值必须在 [0, 255] 范围内。默认值：0。
        - **padding_mode** (:class:`mindspore.dataset.vision.Border` , 可选) - 边界填充方式。可以是 [Border.CONSTANT、Border.EDGE、Border.REFLECT、Border.SYMMETRIC] 中的任何一个。默认值：Border.CONSTANT。

          - **Border.CONSTANT** - 使用常量值进行填充。
          - **Border.EDGE** - 使用各边的边界像素值进行填充。
          - **Border.REFLECT** - 以各边的边界为轴进行镜像填充，忽略边界像素值。
          - **Border.SYMMETRIC** - 以各边的边界为轴进行对称填充，包括边界像素值。

    异常：
        - **TypeError** - 如果 `padding` 不是int或Sequence[int, int], Sequence[int, int, int, int]类型。
        - **TypeError** - 如果 `fill_value` 不是int或tuple[int]类型。
        - **TypeError** - 如果 `padding_mode` 不是 :class:`mindspore.dataset.vision.Border` 的类型。
        - **ValueError** - 如果 `padding` 为负数。
        - **ValueError** - 如果 `fill_value` 不在 [0, 255] 范围内。
        - **RuntimeError** - 如果输入图像的shape不是 <H, W> 或 <H, W, C>。
