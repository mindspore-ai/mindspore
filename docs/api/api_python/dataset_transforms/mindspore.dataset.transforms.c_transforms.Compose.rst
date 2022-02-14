mindspore.dataset.transforms.c_transforms.Compose
=================================================

.. py:class:: mindspore.dataset.transforms.c_transforms.Compose(transforms)

    将多个数据增强算子组合使用。

    **参数：**

    - **transforms** (list) - 一个数据增强的列表。

    **异常：**
      
    - **TypeError** - 参数 `transforms` 类型不为 list。
    - **ValueError** - 参数 `transforms` 的长度为空。
    - **TypeError** - 参数 `transforms` 的元素不是Python可调用对象
      或类型不为 :class:`mindspore.dataset.transforms.c_transforms.TensorOperation` 。
