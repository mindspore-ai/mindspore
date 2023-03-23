mindspore.dataset.vision.AdjustContrast
=======================================

.. py:class:: mindspore.dataset.vision.AdjustContrast(contrast_factor)

    调整输入图像的对比度。

    参数：
        - **contrast_factor** (float) - 对比度调节因子，需为非负数。输入0值将得到灰度图像，1值将得到原始图像，
          2值将调整图像对比度为原来的2倍。

    异常：
        - **TypeError** - 如果 `contrast_factor` 不是float类型。
        - **ValueError** - 如果 `contrast_factor` 小于0。
        - **RuntimeError** - 如果输入图像的形状不是<H, W, C>。
