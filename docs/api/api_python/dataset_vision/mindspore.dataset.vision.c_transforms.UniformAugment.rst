mindspore.dataset.vision.c_transforms.UniformAugment
====================================================

.. py:class:: mindspore.dataset.vision.c_transforms.UniformAugment(transforms, num_ops=2)

    对输入图像执行随机选取的数据增强操作。

    **参数：**

    - **transforms** - C++ 操作列表（不接受Python操作）。
    - **num_ops** (int, optional) - 要选择和执行的操作数量，默认值：2。

    **异常：**

    - **TypeError** - 当 `transforms` 包含不可调用的Python对象。
    - **TypeError** - 当 `num_ops` 不是整数。
    - **ValueError** - 当 `num_ops` 不为正数。
