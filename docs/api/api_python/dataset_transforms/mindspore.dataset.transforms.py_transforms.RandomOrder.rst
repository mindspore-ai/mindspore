mindspore.dataset.transforms.py_transforms.RandomOrder
======================================================

.. py:class:: mindspore.dataset.transforms.py_transforms.RandomOrder(transforms)

    将数据增强列表中的处理随机打乱，然后应用。

    **参数：**

    - **transforms** (list) - 一个数据增强的列表。

    **异常：**
      
    - **TypeError** - 参数 `transforms` 类型不为 list。
    - **TypeError** - 参数 `transforms` 的元素不是Python可调用对象
      或类型不为 :class:`mindspore.dataset.transforms.py_transforms.PyTensorOperation` 。
    - **ValueError** - 参数 `transforms` 的长度为空。
