mindspore.dataset.vision.AutoAugment
====================================

.. py:class:: mindspore.dataset.vision.AutoAugment(policy=AutoAugmentPolicy.IMAGENET, interpolation=Inter.NEAREST, fill_value=0)

    应用AutoAugment数据增强方法，基于论文 `AutoAugment: Learning Augmentation Strategies from Data <https://arxiv.org/pdf/1805.09501.pdf>`_ 。
    此操作仅适用于3通道RGB图像。

    参数：
        - **policy** (:class:`mindspore.dataset.vision.AutoAugmentPolicy` , 可选) - 在不同数据集上学习的AutoAugment策略。默认值：AutoAugmentPolicy.IMAGENET。
          可以是[AutoAugmentPolicy.IMAGENET, AutoAugmentPolicy.CIFAR10, AutoAugmentPolicy.SVHN]中的任何一个。

          - **AutoAugmentPolicy.IMAGENET**：表示应用在ImageNet数据集上学习的AutoAugment。
          - **AutoAugmentPolicy.CIFAR10**：表示应用在Cifar10数据集上学习的AutoAugment。
          - **AutoAugmentPolicy.SVHN**：表示应用在SVHN数据集上学习的AutoAugment。

        - **interpolation** (:class:`mindspore.dataset.vision.Inter` , 可选) - 图像插值方式。默认值：Inter.NEAREST。
          可以是[Inter.NEAREST, Inter.BILINEAR, Inter.BICUBIC, Inter.AREA]中的任何一个。

          - **Inter.NEAREST**：表示插值方法是最近邻插值。
          - **Inter.BILINEAR**：表示插值方法是双线性插值。
          - **Inter.BICUBIC**：表示插值方法为双三次插值。
          - **Inter.AREA**：表示插值方法为像素区域插值。

        - **fill_value** (Union[int, tuple[int]], 可选) - 填充的像素值。
          如果是3元素元组，则分别用于填充R、G、B通道。
          如果是整数，则用于所有 RGB 通道。 `fill_value` 值必须在 [0, 255] 范围内。默认值：0。

    异常：
        - **TypeError** - 如果 `policy` 不是 :class:`mindspore.dataset.vision.AutoAugmentPolicy` 类型。
        - **TypeError** - 如果 `interpolation` 不是 :class:`mindspore.dataset.vision.Inter` 类型。
        - **TypeError** - 如果 `fill_value` 不是整数或长度为3的元组。
        - **RuntimeError** - 如果给定的张量shape不是<H, W, C>。
