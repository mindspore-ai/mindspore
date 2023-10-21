mindspore.dataset.Dataset.apply
===============================

.. py:method:: mindspore.dataset.Dataset.apply(apply_func)

    对数据集对象执行给定操作函数。

    参数：
        - **apply_func** (function) - 数据集处理函数，要求该函数的输入是一个 `Dataset` 对象，返回的是处理后的 `Dataset` 对象。

    返回：
        Dataset，应用了上述操作的新数据集对象。

    异常：
        - **TypeError** - `apply_func` 的类型不是函数。
        - **TypeError** - `apply_func` 未返回 `Dataset` 对象。
