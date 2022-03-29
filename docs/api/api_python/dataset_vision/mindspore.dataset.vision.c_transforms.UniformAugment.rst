mindspore.dataset.vision.c_transforms.UniformAugment
====================================================

.. py:class:: mindspore.dataset.vision.c_transforms.UniformAugment(transforms, num_ops=2)

    对输入图像执行随机选取的数据增强操作。

    **参数：**

    - **transforms** (TensorOperation) - 对给定图像随机选择的边界框区域应用 C++ 变换处理。(不接受Python操作）。
    - **num_ops** (int, 可选) - 要选择和执行的操作的数量，默认值：2。

    **异常：**

    - **TypeError** - 如果 `transform` 不是 :class:`mindspore.dataset.vision.c_transforms` 模块中的图像变换处理。
    - **TypeError** - 当 `num_ops` 不是int类型。
    - **ValueError** - 当 `num_ops` 不为正数。
