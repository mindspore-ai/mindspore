mindspore.dataset.config.get_callback_timeout
===============================================

.. py:function:: mindspore.dataset.config.get_callback_timeout()

    获取 :class:`mindspore.dataset.WaitedDSCallback` 的默认超时时间。
    如果出现死锁，等待的函数将在超时时间结束后退出。

    返回：
        int，表示在出现死锁情况下，用于结束 :class:`mindspore.dataset.WaitedDSCallback` 中的等待函数的超时时间（秒）。
