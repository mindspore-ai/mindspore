mindspore.dataset.vision.Posterize
==================================

.. py:class:: mindspore.dataset.vision.Posterize(bits)

    通过减少输入图像每个颜色通道的位数海报化输入图像。

    参数：
        - **bits** (int) - 每个颜色通道保留的位数，取值需在 [0, 8] 范围内。

    异常：
        - **TypeError** - 如果 `bits` 不是int类型。
        - **ValueError** - 如果 `bits` 不在 [0, 8] 范围内。
        - **RuntimeError** - 如果输入图像的形状不是 <H, W> 或 <H, W, C>。
