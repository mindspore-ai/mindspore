mindspore.hal.is_initialized
=============================

.. py:function:: mindspore.hal.is_initialized(device_target)

    返回目标设备是否已被初始化，此接口将在后续版本中废弃。

    .. note::
        CPU、GPU以及Ascend设备，分别在如下场景被初始化：

        - 分布式任务中，设备会在调用 `mindspore.communication.init` 后初始化。
        - 单卡任务中，会在执行第一个算子或者调用创建流/事件接口后被初始化。

    参数：
        - **device_target** (str) - 目标设备，可选值为 ``"CPU"`` ， ``"GPU"`` 以及 ``"Ascend"`` 。

    返回：
        bool
