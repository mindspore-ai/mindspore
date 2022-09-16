mindspore.dataset.vision.RandomInvert
=====================================

.. py:class:: mindspore.dataset.vision.RandomInvert(prob=0.5)

    以给定的概率随机反转图像的颜色。

    参数：
        - **prob** (float, 可选) - 图像被反转颜色的概率，默认值：0.5。

    异常：
        - **TypeError** - 如果 `prob` 的类型不为bool。
        - **ValueError** - 如果 `prob` 不在 [0, 1] 范围。
        - **RuntimeError** - 如果输入图像的shape不是 <H, W, C>。
