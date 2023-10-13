mindspore.dataset.vision.Invert
===============================

.. py:class:: mindspore.dataset.vision.Invert()

    对输入的RGB图像进行色彩反转。

    对于图像中的每个像素，若原像素值为 `pixel` ，则反转后的像素值为 `255 - pixel` 。

    异常：
        - **RuntimeError** - 如果输入图像的shape不是 <H, W, C>。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_
