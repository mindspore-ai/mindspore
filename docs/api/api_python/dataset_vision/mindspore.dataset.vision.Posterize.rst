mindspore.dataset.vision.Posterize
==================================

.. py:class:: mindspore.dataset.vision.Posterize(bits)

    减少图像的颜色通道的比特位数，使图像变得高对比度和颜色鲜艳，类似于海报或印刷品的效果。

    参数：
        - **bits** (int) - 每个颜色通道保留的位数，取值需在 [0, 8] 范围内。

    异常：
        - **TypeError** - 如果 `bits` 不是int类型。
        - **ValueError** - 如果 `bits` 不在 [0, 8] 范围内。
        - **RuntimeError** - 如果输入图像的形状不是 <H, W> 或 <H, W, C>。
