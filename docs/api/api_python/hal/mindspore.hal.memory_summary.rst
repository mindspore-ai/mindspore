mindspore.hal.memory_summary
============================

.. py:function:: mindspore.hal.memory_summary(device_target=None)

    返回可读的内存池状态信息。

    .. note::
        - 若用户不指定 `device_target` ，将此参数设置为当前已经设置的后端类型。

    参数：
        - **device_target** (str，可选) - 用户指定的后端类型，必须是 ``"CPU"`` ， ``"GPU"`` 以及 ``"Ascend"`` 的其中一个。默认值：``None``。

    返回：
        str，表格形式的可读内存池状态信息。
