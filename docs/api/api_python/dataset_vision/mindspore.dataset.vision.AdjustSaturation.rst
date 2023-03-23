mindspore.dataset.vision.AdjustSaturation
=========================================

.. py:class:: mindspore.dataset.vision.AdjustSaturation(saturation_factor)

    调整输入图像的饱和度。

    输入图像的shape需为 [H, W, C]。

    参数：
        - **saturation_factor** (float) - 饱和度调节因子，需为非负数。输入0值将得到全黑图像，1值将得到原始图像，
          2值将调整图像饱和度为原来的2倍。

    异常：
        - **TypeError** - 如果 `saturation_factor` 不是float类型。
        - **ValueError** - 如果 `saturation_factor` 小于0。
        - **RuntimeError** - 如果输入图像的形状不是<H, W, C>。
        - **RuntimeError** - 如果输入图像的通道数不是3。
