mindspore.dataset.vision.c_transforms.Invert
============================================

.. py:class:: mindspore.dataset.vision.c_transforms.Invert()

    在 RGB 模式下对输入图像应用像素反转, 计算方式为（255 - pixel）。

    **异常：**

    - **RuntimeError** - 如果输入图像的shape不是 <H, W, C>。
