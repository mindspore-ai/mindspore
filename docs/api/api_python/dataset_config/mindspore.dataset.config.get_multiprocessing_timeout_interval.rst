mindspore.dataset.config.get_multiprocessing_timeout_interval
==============================================================

.. py:function:: mindspore.dataset.config.get_multiprocessing_timeout_interval()

    获取在多进程/多线程下，主进程/主线程获取数据超时时，告警日志打印的时间间隔的全局配置。

    返回：
        int，表示多进程/多线程下，主进程/主线程获取数据超时时，告警日志打印的时间间隔（默认300秒）。
