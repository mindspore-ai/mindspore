mindspore.dataset.vision.FiveCrop
=================================

.. py:class:: mindspore.dataset.vision.FiveCrop(size)

    在输入PIL图像的中心与四个角处分别裁剪指定尺寸大小的子图。

    参数：
        - **size** (Union[int, Sequence[int, int]]) - 裁剪子图的尺寸大小。若输入int，则以该值为边长裁剪( `size` , `size` )尺寸大小的子图；若输入Sequence[int, int]，则以2个元素分别为高和宽裁剪子图。
    
    异常：
        - **TypeError** - 当 `size` 的类型不为int或Sequence[int, int]。
        - **ValueError** - 当 `size` 不为正数。
