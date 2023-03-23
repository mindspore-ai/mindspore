mindspore.dataset.vision.AdjustBrightness
=========================================

.. py:class:: mindspore.dataset.vision.AdjustBrightness(brightness_factor)

    调整输入图像的亮度。

    参数：
        - **brightness_factor** (float) - 亮度调节因子，需为非负数。输入0值将得到全黑图像，1值将得到原始图像，
          2值将调整图像亮度为原来的2倍。

    异常：
        - **TypeError** - 如果 `brightness_factor` 不是float类型。
        - **ValueError** - 如果 `brightness_factor` 小于0。
        - **RuntimeError** - 如果输入图像的形状不是<H, W, C>。
