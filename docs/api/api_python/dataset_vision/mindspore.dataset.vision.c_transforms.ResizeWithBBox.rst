mindspore.dataset.vision.c_transforms.ResizeWithBBox
====================================================

.. py:class:: mindspore.dataset.vision.c_transforms.ResizeWithBBox(size, interpolation=Inter.LINEAR)

    将输入图像调整为给定的大小并相应地调整边界框的大小。

    **参数：**

    - **size** (Union[int, sequence]) - 图像的输出大小。如果 `size` 是整数，将调整图像的较短边长度为 `size`，且保持图像的宽高比不变；若输入是2元素组成的序列，其输入格式需要是 (height, width) 。
    - **interpolation** (Inter mode, optional) - 图像插值模式。它可以是 [Inter.LINEAR, Inter.NEAREST, Inter.BICUBIC, Inter.PILCUBIC] 中的任何一个，默认值：Inter.LINEAR。

      - Inter.LINEAR，双线性插值。
      - Inter.NEAREST，最近邻插值。
      - Inter.BICUBIC，双三次插值。

    **异常：**

    - **TypeError** - 当 `size` 的类型不为整型或整型序列。
    - **TypeError** - 当 `interpolation` 的类型不为 :class:`Inter` 。
    - **ValueError** - 当 `size` 不为正数。
    - **RuntimeError** - 如果输入的Tensor不是 <H, W> 或 <H, W, C> 格式。
