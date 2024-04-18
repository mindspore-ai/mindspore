mindspore.dataset.vision.AutoAugment
====================================

.. py:class:: mindspore.dataset.vision.AutoAugment(policy=AutoAugmentPolicy.IMAGENET, interpolation=Inter.NEAREST, fill_value=0)

    应用AutoAugment数据增强方法，基于论文 `AutoAugment: Learning Augmentation Strategies from Data <https://arxiv.org/pdf/1805.09501.pdf>`_ 。
    此操作仅适用于3通道RGB图像。

    参数：
        - **policy** (:class:`~.vision.AutoAugmentPolicy`, 可选) - 在不同数据集上学习的AutoAugment策略。默认值： ``AutoAugmentPolicy.IMAGENET`` 。
          可以是 ``AutoAugmentPolicy.IMAGENET`` 、 ``AutoAugmentPolicy.CIFAR10`` 、 ``AutoAugmentPolicy.SVHN`` 。

          - **AutoAugmentPolicy.IMAGENET**：表示应用在ImageNet数据集上学习的AutoAugment。
          - **AutoAugmentPolicy.CIFAR10**：表示应用在Cifar10数据集上学习的AutoAugment。
          - **AutoAugmentPolicy.SVHN**：表示应用在SVHN数据集上学习的AutoAugment。

        - **interpolation** (:class:`~.vision.Inter`, 可选) - 图像插值方法。可选值详见 :class:`mindspore.dataset.vision.Inter` 。
          默认值： ``Inter.NEAREST``。
        - **fill_value** (Union[int, tuple[int]], 可选) - 填充的像素值。
          如果是3元素元组，则分别用于填充R、G、B通道。
          如果是整数，则用于所有 RGB 通道。 `fill_value` 值必须在 [0, 255] 范围内。默认值： ``0`` 。

    异常：
        - **TypeError** - 如果 `policy` 不是 :class:`mindspore.dataset.vision.AutoAugmentPolicy` 类型。
        - **TypeError** - 如果 `interpolation` 不是 :class:`mindspore.dataset.vision.Inter` 类型。
        - **TypeError** - 如果 `fill_value` 不是整数或长度为3的元组。
        - **RuntimeError** - 如果给定的张量shape不是<H, W, C>。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_
