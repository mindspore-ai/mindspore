mindspore.dataset.config.set_monitor_sampling_interval
=======================================================

.. py:function:: mindspore.dataset.config.set_monitor_sampling_interval(interval)

    设置性能监测采样的默认间隔时间（毫秒）。

    参数：
        - **interval** (int) - 表示用于性能监测采样的间隔时间（毫秒）。

    异常：
        - **TypeError** - `interval` 不是int类型。
        - **ValueError** - `interval` 小于等于0或大于 `INT32_MAX(2147483647)`。
