mindspore.dataset.vision.c_transforms.RandomHorizontalFlipWithBBox
==================================================================

.. py:class:: mindspore.dataset.vision.c_transforms.RandomHorizontalFlipWithBBox(prob=0.5)

    以给定的概率水平随机翻转输入图像并相应地调整边界框。

    **参数：**

    - **prob**  (float, 可选) - 图像被翻转的概率，必须在 [0, 1] 范围内, 默认值：0.5。

    **异常：**

    - **TypeError** - 如果 `prob` 不是float类型。
    - **ValueError** - 如果 `prob` 不在 [0, 1] 范围内。
    - **RuntimeError** - 如果输入图像的shape不是 <H, W> 或 <H, W, C>。
