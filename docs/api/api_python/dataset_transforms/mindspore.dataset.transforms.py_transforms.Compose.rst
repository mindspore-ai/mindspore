mindspore.dataset.transforms.py_transforms.Compose
==================================================

.. py:class:: mindspore.dataset.transforms.py_transforms.Compose(transforms)

    将多个数据增强算子组合使用。

    .. note::
        Compose可以将 `mindspore.dataset.transforms.py_transforms` 模块中的数据增强算子以及用户自定义的Python可调用对象
        合并成单个数据增强。对于用户定义的Python可调用对象，要求其返回值是numpy.ndarray类型。有关如何使用，请参阅Compose的示例，或阅读
        :class:`mindspore.dataset.vision.py_transforms.FiveCrop` 的示例，学习如何与用户自定义Python可调用对象配合使用。

    **参数：**

    - **transforms** (list) - 一个数据增强的列表。

    **异常：**

    - **TypeError** - 参数 `transforms` 类型不为 list。
    - **ValueError** - 参数 `transforms` 的长度为空。
    - **TypeError** - 参数 `transforms` 的元素不是Python的callable对象。

    .. py:method:: reduce(operations)

        使用Compose将指定数据增强操作列表中相邻的Python操作组合，以允许混用Python和C++操作。

        **参数：**

        - **operations** (list) - 数据增强的列表。

        **返回：**

        list，组合后的数据增强操作列表。
