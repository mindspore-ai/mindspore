mindspore.dataset.vision.py_transforms.CenterCrop
=================================================

.. py:class:: mindspore.dataset.vision.py_transforms.CenterCrop(size)

    以输入PIL图像的中心为裁剪中心，裁剪指定大小的子图。

    **参数：**

    - **size** (Union[int, sequence]) - 裁剪子图的大小。若输入整型，则以该值为边裁剪(size, size)大小的子图；若输入2元素序列，则以2个元素分别为高和宽裁剪(height, width)大小的子图。

    **异常：**

    - **TypeError** - 当 `size` 的类型不为整型或整型序列。
    - **ValueError** - 当 `size` 小于等于0。
