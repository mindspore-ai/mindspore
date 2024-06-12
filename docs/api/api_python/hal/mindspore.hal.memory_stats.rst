mindspore.hal.memory_stats
==========================

.. py:function:: mindspore.hal.memory_stats(device_target=None)

    返回从内存池查询到的状态信息。

    .. note::
        - 若用户不指定 `device_target` ，将此参数设置为当前已经设置的后端类型。
        - 对于  `CPU` 后端，固定返回数据为空的字典。

    参数：
        - **device_target** (str，可选) - 用户指定的后端类型，必须是 ``"CPU"`` ， ``"GPU"`` 以及 ``"Ascend"`` 的其中一个。默认值：``None``。

    返回：
        dict，查询到的内存信息。
