mindspore.dataset.vision.AdjustSharpness
========================================

.. py:class:: mindspore.dataset.vision.AdjustSharpness(sharpness_factor)

    调整输入图像的锐度。

    输入图像的shape需为 [H, W, C] 或 [H, W]。

    参数：
        - **sharpness_factor** (float) - 锐度调节因子，需为非负数。输入0值将得到模糊图像，1值将得到原始图像，
          2值将调整图像锐度为原来的2倍。

    异常：
        - **TypeError** - 如果 `sharpness_factor` 不是float类型。
        - **ValueError** - 如果 `sharpness_factor` 小于0。
        - **RuntimeError** - 如果输入图像的形状不是<H, W, C>或<H, W>。
