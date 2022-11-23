mindspore.dataset.config.get_monitor_sampling_interval
=======================================================

.. py:function:: mindspore.dataset.config.get_monitor_sampling_interval()

    获取性能监控采样时间间隔的全局配置。
    如果 `set_monitor_sampling_interval` 方法未被调用，那么将会返回默认值1000。

    返回：
        int，表示性能监控采样间隔时间（毫秒）。
