mindspore.dataset.config.set_enable_shared_mem
===============================================

.. py:function:: mindspore.dataset.config.set_enable_shared_mem(enable)

    设置是否在开启数据处理多进程时使用共享内存进行进程间通信。

    使用共享内存可以加速进程间的数据传递效率。

    该功能默认开启。

    .. note::
        - 暂不支持Windows和MacOS系统。
        - `set_enable_shared_mem` 接口将在后续版本废弃。

    参数：
        - **enable** (bool) - 是否使用共享内存进行进程间通信。

    异常：
        - **TypeError** - 当 `enable` 不为bool类型。
