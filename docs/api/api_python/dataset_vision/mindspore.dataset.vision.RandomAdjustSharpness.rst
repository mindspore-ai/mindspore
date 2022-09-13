mindspore.dataset.vision.RandomAdjustSharpness
==============================================

.. py:class:: mindspore.dataset.vision.RandomAdjustSharpness(degree, prob=0.5)

    以给定的概率随机调整输入图像的清晰度。

    参数：
        - **degrees** (float) - 锐度调整度，必须是非负的。
          0.0度表示模糊图像，1.0度表示原始图像，2.0度表示清晰度增加2倍。
        - **prob** (float, 可选) - 图像被锐化的概率，默认值：0.5。

    异常：
        - **TypeError** - 如果 `degree` 的类型不为float。
        - **TypeError** - 如果 `prob` 的类型不为bool。
        - **ValueError** - 如果 `prob` 不在 [0, 1] 范围。
        - **ValueError** - 如果 `degree` 为负数。
        - **RuntimeError** -如果给定的张量形状不是<H, W>或<H, W, C>。
