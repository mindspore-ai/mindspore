mindspore.dataset.transforms.c_transforms.OneHot
================================================

.. py:class:: mindspore.dataset.transforms.c_transforms.OneHot(num_classes)

    将Tensor进行OneHot编码。

    **参数：**

    - **num_classes** (int) - 数据集的类别数，它应该大于数据集中最大的label编号。

    **异常：**
      
    - **TypeError** - 参数 `num_classes` 类型不为 int。
    - **RuntimeError** - 输入Tensor的类型不为 int。
    - **RuntimeError** - 参数Tensor的shape不是1-D。
