mindspore.dataset.vision.py_transforms.TenCrop
==============================================

.. py:class:: mindspore.dataset.vision.py_transforms.TenCrop(size, use_vertical_flip=False)

    在输入PIL图像的中心与四个角处分别裁剪指定大小的子图，并将其翻转图一并返回。

    **参数：**

    - **size** (Union[int, sequence]) - 裁剪子图的大小。若输入整型，则以该值为边裁剪(size, size)大小的子图；若输入2元素序列，则以2个元素分别为高和宽裁剪(height, width)大小的子图。
    - **use_vertical_flip** (bool，可选) - 若为True，将对子图进行垂直翻转；否则进行水平翻转。默认值：False。

    **异常：**
        
    - **TypeError** - 当 `size` 的类型不为整型或整型序列。
    - **TypeError** - 当 `use_vertical_flip` 的类型不为布尔型。
    - **ValueError** - 当 `size` 不为正数。
