mindspore.dataset.transforms.Compose
====================================

.. py:class:: mindspore.dataset.transforms.Compose(transforms)

    将多个数据增强操作组合使用。

    .. note::
        Compose可以将 `mindspore.dataset.transforms` / `mindspore.dataset.vision` 等模块中的数据增强操作以及用户自定义的Python可调用对象
        合并成单个数据增强。对于用户定义的Python可调用对象，要求其返回值是numpy.ndarray类型。

    参数：
        - **transforms** (list) - 一个数据增强的列表。

    异常：
        - **TypeError** - 参数 `transforms` 类型不为list。
        - **ValueError** - 参数 `transforms` 是空的list。
        - **TypeError** - 参数 `transforms` 的元素不是Python的可调用对象或audio/text/transforms/vision模块中的数据增强方法。

    .. py:method:: decompose(operations)

        从给定的操作列表中删除所有 compose 操作。

        参数：
            - **operations** (list) - 变换列表。

        返回：
            没有组合操作的操作列表。

    .. py:method:: reduce(operations)

        在 Compose 中包装相邻的 Python 操作，以允许混合 Python 和 C++ 操作。

        参数：
            - **operations** (list) - Tensor操作列表。

        返回：
            list，简化的操作列表。
