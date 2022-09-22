mindspore.dataset.vision.RandomLighting
========================================

.. py:class:: mindspore.dataset.vision.RandomLighting(alpha=0.05)

    将AlexNet PCA的噪声添加到图像中。Alexnet PCA噪声的特征值和特征向量是由ImageNet数据集计算得出。

    参数：
        - **alpha** (float, 可选) - 图像的强度，必须是非负的。默认值：0.05。

    异常：
        - **TypeError** - 如果 `alpha` 的类型不为bool。
        - **ValueError** - 如果 `alpha` 为负数。
        - **RuntimeError** - 如果输入图像的shape不是 <H, W, C>。
