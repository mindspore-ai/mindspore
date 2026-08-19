mindspore.dataset.transforms.RandomApply
========================================

.. py:class:: mindspore.dataset.transforms.RandomApply(transforms, prob=0.5)

    按给定概率随机执行一组变换。

    参数：
        - **transforms** (list) - 一个数据增强的列表。
        - **prob** (float, 可选) - 随机应用变换列表的概率，取值范围：[0.0, 1.0]。默认值： ``0.5`` 。

    异常：
        - **TypeError** - 参数 `transforms` 类型不为list。
        - **ValueError** - 参数 `transforms` 的长度为空。
        - **TypeError** - 参数 `transforms` 的元素不是Python可调用对象或audio/text/transforms/vision模块中的数据处理操作。
        - **TypeError** - 参数 `prob` 的类型不为float。
        - **ValueError** - 参数 `prob` 的取值范围不为[0.0, 1.0]。
