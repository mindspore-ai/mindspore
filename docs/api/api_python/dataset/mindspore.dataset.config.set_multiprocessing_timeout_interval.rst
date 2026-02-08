mindspore.dataset.config.set_multiprocessing_timeout_interval
==============================================================

.. py:function:: mindspore.dataset.config.set_multiprocessing_timeout_interval(interval)

    设置在多进程/多线程下，主进程/主线程获取数据超时时，告警日志打印的默认时间间隔（秒）。

    参数：
        - **interval** (int) - 表示多进程/多线程下，主进程/主线程获取数据超时时，告警日志打印的时间间隔（秒）。

    异常：
        - **TypeError** - `interval` 不是int类型。
        - **ValueError** - `interval` 小于等于0或大于 `INT32_MAX(2147483647)`。
