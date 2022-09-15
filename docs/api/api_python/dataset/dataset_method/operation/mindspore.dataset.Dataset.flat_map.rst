mindspore.dataset.Dataset.flat_map
==================================

.. py:method:: mindspore.dataset.Dataset.flat_map(func)

    对数据集对象中每一条数据执行给定的数据处理，并将结果展平。

    参数：
        - **func** (function) - 数据处理函数，要求输入必须为一个 `numpy.ndarray` ，返回值是一个 `Dataset` 对象。

    返回：
        执行给定操作后的数据集对象。

    异常：
        - **TypeError** - `func` 不是函数。
        - **TypeError** - `func` 的返回值不是 `Dataset` 对象。
