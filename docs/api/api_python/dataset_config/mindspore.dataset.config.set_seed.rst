mindspore.dataset.config.set_seed
==================================

.. py:function:: mindspore.dataset.config.set_seed(seed)

    设置随机种子，产生固定的随机数来达到确定的结果。

    .. note::
        此函数在Python随机库和numpy.random库中设置种子，以便随机进行确定性Python增强。此函数应与创建的每个迭代器一起调用，以重置随机种子。

    参数：
        - **seed** (int) - 表示随机数量的种子。该参数用于生成确定性随机数。

    异常：
        - **TypeError** - `seed` 不是int类型。
        - **ValueError** - `seed` 小于0或 `seed` 大于 `UINT32_MAX(4294967295)` 时， `seed` 无效。
