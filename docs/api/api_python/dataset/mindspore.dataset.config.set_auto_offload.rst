mindspore.dataset.config.set_auto_offload
===============================================

.. py:function:: mindspore.dataset.config.set_auto_offload(offload)

    设置是否开启自动卸载功能。

    自动卸载功能可以自动将数据处理的部分运算分配到不同的异构硬件（GPU或Ascend）上，以提高数据处理的速度。

    参数：
        - **offload** (bool) - 是否开启自动卸载功能。

    异常：
        - **TypeError** - 当 `offload` 的类型不为bool。
