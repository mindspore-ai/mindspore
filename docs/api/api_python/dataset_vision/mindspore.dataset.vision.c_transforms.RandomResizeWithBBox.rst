mindspore.dataset.vision.c_transforms.RandomResizeWithBBox
==========================================================

.. py:class:: mindspore.dataset.vision.c_transforms.RandomResizeWithBBox(size)

    使用随机选择的插值模式来调整输入图像的大小，并相应地调整边界框的大小。

    **参数：**

    - **size** (Union[int, sequence]) - 调整后图像的输出大小。如果 `size` 是一个整数，图像的短边将被调整为 `size` 大小，并依据短边的调整比例相应调整图像长边的大小。如果 `size` 是一个长度为2的序列，其输入格式应该为 (height, width)。
