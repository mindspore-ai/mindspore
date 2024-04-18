mindspore.dataset.vision.RandomColor
====================================

.. py:class:: mindspore.dataset.vision.RandomColor(degrees=(0.1, 1.9))

    随机调整输入图像的颜色。此操作仅适用于 3 通道RGB图像。

    参数：
        - **degrees** (Sequence[float], 可选) - 色彩调节系数的范围，必须为非负数。它应该是(min, max)格式。
          如果min与max相等，则代表色彩变化步长固定。默认值： ``(0.1, 1.9)`` 。

    异常：
        - **TypeError** - 如果 `degrees` 不是Sequence[float]类型。
        - **ValueError** - 如果 `degrees` 为负数。
        - **RuntimeError** - 如果输入图像的shape不是 <H, W, C>。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_
