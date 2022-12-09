mindspore.dataset.config.set_callback_timeout
===============================================

.. py:function:: mindspore.dataset.config.set_callback_timeout(timeout)

    为 :class:`mindspore.dataset.WaitedDSCallback` 设置的默认超时时间（秒）。

    参数：
        - **timeout** (int) - 表示在出现死锁情况下，用于结束 :class:`mindspore.dataset.WaitedDSCallback` 中等待的超时时间（秒）。 `timeout` 取值必须大于0。

    异常：
        - **TypeError** - `timeout` 不是int类型。
        - **ValueError** - 如果 `timeout` 不为正数。
