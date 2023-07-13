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