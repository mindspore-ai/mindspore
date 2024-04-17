mindspore.dataset.transforms.RandomChoice
=========================================

.. py:class:: mindspore.dataset.transforms.RandomChoice(transforms)

    从一组数据增强变换中随机选择一个进行应用。

    参数：
        - **transforms** (list) - 可供选择的数据增强变换列表。

    异常：
        - **TypeError** - 参数 `transforms` 类型不为list。
        - **ValueError** - 参数 `transforms` 为空。
        - **TypeError** - 参数 `transforms` 的元素不是Python可调用对象或audio/text/transforms/vision模块中的数据处理操作。
