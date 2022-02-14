mindspore.dataset.transforms.c_transforms.RandomChoice
======================================================

.. py:class:: mindspore.dataset.transforms.c_transforms.RandomChoice(transforms)

    在一组数据增强中随机选择部分增强处理进行应用。

    **参数：**

    - **transforms** (list) - 一个数据增强的列表。

    **异常：**
      
    - **TypeError** - 参数 `transforms` 类型不为 list。
    - **ValueError** - 参数 `transforms` 的长度为空。
    - **TypeError** - 参数 `transforms` 的元素不是Python可调用对象
      或类型不为 :class:`mindspore.dataset.transforms.c_transforms.TensorOperation` 。
