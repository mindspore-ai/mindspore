mindspore.dataset.vision.AdjustHue
==================================

.. py:class:: mindspore.dataset.vision.AdjustHue(hue_factor)

    调整输入图像的色调。

    参数：
        - **hue_factor** (float) - 色调通道调节值，值必须在 [-0.5, 0.5] 范围内。

    异常：
        - **TypeError** - 如果 `hue_factor` 不是float类型。
        - **ValueError** - 如果 `hue_factor` 不在[-0.5, 0.5]的范围内。
        - **RuntimeError** - 如果输入图像的形状不是<H, W, C>。
