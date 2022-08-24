mindspore.dataset.vision.Invert
===============================

.. py:class:: mindspore.dataset.vision.Invert()

    在 RGB 模式下对输入图像应用像素反转。将每一个像素重新赋值为（255 - pixel）。

    异常：
        - **RuntimeError** - 如果输入图像的shape不是 <H, W, C>。
