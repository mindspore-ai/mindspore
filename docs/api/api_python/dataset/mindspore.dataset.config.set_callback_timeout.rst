mindspore.dataset.config.set_callback_timeout
===============================================

.. py:function:: mindspore.dataset.config.set_callback_timeout(timeout)

    为 :class:`mindspore.dataset.WaitedDSCallback` 设置的默认超时时间（秒）。

    参数：
        - **timeout** (int) - 表示在出现死锁情况下，用于结束 :class:`mindspore.dataset.WaitedDSCallback` 中等待的超时时间（秒）。

    异常：
        - **TypeError** - `timeout` 不是int类型。
        - **ValueError** - `timeout` 小于等于0或 `timeout` 大于 `INT32_MAX(2147483647)` 时 `timeout` 无效。
