mindspore.dataset.vision.c_transforms.RandomVerticalFlipWithBBox
=================================================

.. py:class:: mindspore.dataset.vision.c_transforms.RandomVerticalFlipWithBBox(prob=0.5)

    以给定的概率对输入图像和边界框在垂直方向进行随机翻转。

    **参数：**

    - **prob** (float, optional) - 图像被翻转的概率，默认值：0.5。

    **异常：**

    - **TypeError** - 如果 `prob` 不是float类型。
    - **ValueError** - 如果 `prob` 不在 [0, 1] 范围。
    - **RuntimeError** - 如果输入的Tensor不是 <H, W> 或<H, W, C> 格式。
