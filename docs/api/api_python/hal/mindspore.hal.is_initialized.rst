mindspore.hal.is_initialized
=============================

.. py:function:: mindspore.hal.is_initialized(device_target)

    返回指定后端是否已被初始化。

    .. note::
        CPU、GPU以及Ascend后端，分别为在如下场景被初始化：
        - 分布式任务中，后端会在调用 `mindspore.communication.init` 后初始化。
        - 单卡任务中，会在执行第一个算子或者调用创建流/事件接口后被初始化。

    参数：
        - **device_target** (str) - 用户指定的后端类型，必须是 ``"CPU"`` ， ``"GPU"`` 以及 ``"Ascend"`` 的其中一个。

    返回：
        bool，指定后端是否已被初始化。
