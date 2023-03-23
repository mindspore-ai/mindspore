mindspore.dataset.vision.PadToSize
==================================

.. py:class:: mindspore.dataset.vision.PadToSize(size, offset=None, fill_value=0, padding_mode=Border.CONSTANT)

    将图像填充到固定大小。

    参数：
        - **size** (Union[int, Sequence[int, int]]) - 要填充的目标大小。
          若输入整型，则将图像填充为(size, size)大小；如果提供了序列[int, int]，则将图像填充为(高度, 宽度)大小。
        - **offset** (Union[int, Sequence[int, int]], 可选) - 顶部和左侧要填充的长度。
          如果输入整型，使用此值填充图像上侧和左侧。
          如果提供了序列[int, int]，则应按[top, left]的顺序排列，填充图像上侧和左侧。
          默认值：None，表示对称填充，保持原始图像处于中心位置。
        - **fill_value** (Union[int, tuple[int, int, int]], 可选) - 填充的像素值，仅在 `padding_mode` 取值为Border.CONSTANT时有效。
          如果是3元素元组，则分别用于填充R、G、B通道。
          如果是整数，则用于所有 RGB 通道。
          `fill_value` 值必须在 [0, 255] 范围内。默认值：0。
        - **padding_mode** (:class:`mindspore.dataset.vision.Border` , 可选) - 边界填充方式。可以是 [Border.CONSTANT、Border.EDGE、Border.REFLECT、Border.SYMMETRIC] 中的任何一个。默认值：Border.CONSTANT。

          - **Border.CONSTANT** - 使用常量值进行填充。
          - **Border.EDGE** - 使用各边的边界像素值进行填充。
          - **Border.REFLECT** - 以各边的边界为轴进行镜像填充，忽略边界像素值。
          - **Border.SYMMETRIC** - 以各边的边界为轴进行对称填充，包括边界像素值。

    异常：
        - **TypeError** - 如果 `size` 不是int或tuple[int, int]类型。
        - **TypeError** - 如果 `offset` 不是int或tupl[int, int]类型。
        - **TypeError** - 如果 `fill_value` 不是int或tuple[int]类型。
        - **TypeError** - 如果 `padding_mode` 不是 :class:`mindspore.dataset.vision.Border` 的类型。
        - **ValueError** - 如果 `size` 不是正数。
        - **ValueError** - 如果 `offset` 为负数。
        - **ValueError** - 如果 `fill_value` 不在[0, 255]的范围内。
        - **RuntimeError** - 如果输入图像的形状不是<H, W>或<H, W, C>。
