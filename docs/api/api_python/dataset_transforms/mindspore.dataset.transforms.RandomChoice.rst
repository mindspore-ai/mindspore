mindspore.dataset.transforms.RandomChoice
=========================================

.. py:class:: mindspore.dataset.transforms.RandomChoice(transforms)

    在一组数据增强中随机选择部分增强处理进行应用。

    参数：
        - **transforms** (list) - 一个数据增强的列表。

    异常：
        - **TypeError** - 参数 `transforms` 类型不为list。
        - **ValueError** - 参数 `transforms` 是空的list。
        - **TypeError** - 参数 `transforms` 的元素不是Python可调用对象或audio/text/transforms/vision模块中的数据处理操作。
