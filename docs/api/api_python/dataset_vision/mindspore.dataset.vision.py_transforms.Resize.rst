mindspore.dataset.vision.py_transforms.Resize
=============================================

.. py:class:: mindspore.dataset.vision.py_transforms.Resize(size, interpolation=Inter.BILINEAR)

    将输入PIL图像放缩为指定尺寸大小。

    **参数：**

    - **size** (Union[int, Sequence[int, int]]) - 图像放缩的尺寸大小。若输入int，将调整图像的较短边为此值，而保持图像的宽高比不变；若输入Sequence[int, int]，则以2个元素分别为高和宽进行放缩。
    - **interpolation** (Inter，可选) - 插值方式，取值可为 Inter.NEAREST、Inter.ANTIALIAS、Inter.BILINEAR 或 Inter.BICUBIC。默认值：Inter.BILINEAR。

      - **Inter.NEAREST**：最近邻插值。
      - **Inter.ANTIALIAS**：抗锯齿插值。
      - **Inter.BILINEAR**：双线性插值。
      - **Inter.BICUBIC**：双三次插值。

    **异常：**

    - **TypeError** - 当 `size` 的类型不为int或Sequence[int, int]。
    - **TypeError** - 当 `interpolation` 的类型不为 :class:`mindspore.dataset.vision.Inter` 。
    - **ValueError** - 当 `size` 不为正数。
