mindspore.dataset.config.set_enable_watchdog
===============================================

.. py:function:: mindspore.dataset.config.set_enable_watchdog(enable)

    设置watchdog Python线程是否启用。默认情况下，watchdog Python线程是启用的。watchdog Python线程负责清理卡死或假死的子进程。

    参数：
        - **enable** (bool) - 是否开启watchdog Python线程。

    异常：
        - **TypeError** - `enable` 不是bool类型。
