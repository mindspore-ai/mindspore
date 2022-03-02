mindspore.dataset.transforms.py_transforms.RandomChoice
=======================================================

.. py:class:: mindspore.dataset.transforms.py_transforms.RandomChoice(transforms)

    在一组数据增强中随机选择部分增强处理进行应用。

    **参数：**

    - **transforms** (list) - 一个数据增强的列表。

    **异常：**
      
    - **TypeError** - 参数 `transforms` 类型不为 list。
    - **TypeError** - 参数 `transforms` 的元素不是Python可调用对象或py_transforms模块中的数据处理操作。
    - **ValueError** - 参数 `transforms` 的长度为空。
