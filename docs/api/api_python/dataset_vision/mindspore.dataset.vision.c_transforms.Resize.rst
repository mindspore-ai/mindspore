mindspore.dataset.vision.c_transforms.Resize
============================================

.. py:class:: mindspore.dataset.vision.c_transforms.Resize(size, interpolation=Inter.LINEAR)

    对输入图像使用给定的 :class:`mindspore.dataset.vision.Inter` 插值方式去调整为给定的尺寸大小。

    **参数：**

    - **size** (Union[int, Sequence[int]]) - 图像的输出尺寸大小。若输入整型，则放缩至(size, size)大小；若输入2元素序列，则以2个元素分别为高和宽放缩至(高度, 宽度)大小。
    - **interpolation** (Inter, 可选) - 图像插值方式。它可以是 [Inter.LINEAR, Inter.NEAREST, Inter.BICUBIC, Inter.PILCUBIC] 中的任何一个，默认值：Inter.LINEAR。

      - Inter.LINEAR，双线性插值。
      - Inter.NEAREST，最近邻插值。
      - Inter.BICUBIC，双三次插值。
      - Inter.AREA，像素区域插值。
      - Inter.PILCUBIC，Pillow库中实现的双三次插值，输入应为3通道格式。

    **异常：**

    - **TypeError** - 当 `size` 的类型不为int或Sequence[int]。
    - **TypeError** - 当 `interpolation` 的类型不为 :class:`mindspore.dataset.vision.Inter` 。
    - **ValueError** - 当 `size` 不为正数。
    - **RuntimeError** - 如果输入的Tensor不是 <H, W> 或 <H, W, C> 格式。
