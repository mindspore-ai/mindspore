mindspore.dataset.vision.c_transforms.BoundingBoxAugment
========================================================

.. py:class:: mindspore.dataset.vision.c_transforms.BoundingBoxAugment(transform, ratio=0.3)

    对图像的标注边界框(bounding box)区域随机应用给定的图像变换处理。

    **参数：**

    - **transform** (TensorOperation) - 要应用的图像变换处理。
    - **ratio**  (float, 可选) - 应用图像变换处理的概率。范围：[0, 1], 默认值：0.3。

    **异常：**

    - **TypeError** - 如果 `transform` 不是 :class:`mindspore.dataset.vision.c_transforms` 模块中的图像处理操作。
    - **TypeError** - 如果 `ratio` 不是float类型。
    - **ValueError** - 如果 `ratio` 不在 [0, 1] 范围内。
    - **RuntimeError** - 如果给定的边界框无效。
