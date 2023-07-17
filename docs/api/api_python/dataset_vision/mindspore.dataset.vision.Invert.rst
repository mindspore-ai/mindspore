mindspore.dataset.vision.Invert
===============================

.. py:class:: mindspore.dataset.vision.Invert()

    在 RGB 模式下对输入图像应用像素反转。将每一个像素重新赋值为（255 - pixel）。

    异常：
        - **RuntimeError** - 如果输入图像的shape不是 <H, W, C>。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/r2.1/api_python/samples/dataset/vision_gallery.html>`_
