mindspore.dataset.vision.RandomEqualize
=======================================

.. py:class:: mindspore.dataset.vision.RandomEqualize(prob=0.5)

    以给定的概率随机对输入图像进行直方图均衡化。

    参数：
        - **prob** (float, 可选) - 图像被均衡化的概率，取值范围：[0.0, 1.0]。默认值：0.5。

    异常：
        - **TypeError** - 如果 `prob` 的类型不为float。
        - **ValueError** - 如果 `prob` 不在 [0.0, 1.0] 范围。
        - **RuntimeError** - 如果输入图像的shape不是 <H, W> 或 <H, W, C>。
