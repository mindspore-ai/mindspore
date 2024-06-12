mindspore.hal.max_memory_reserved
=================================

.. py:function:: mindspore.hal.max_memory_reserved(device_target=None)

    返回从进程启动开始，内存池管理的内存总量的峰值。

    .. note::
        - 若用户不指定 `device_target` ，将此参数设置为当前已经设置的后端类型。
        - 对于 `CPU` 后端，固定返回0。

    参数：
        - **device_target** (str，可选) - 用户指定的后端类型，必须是 ``"CPU"`` ， ``"GPU"`` 以及 ``"Ascend"`` 的其中一个。默认值：``None``。

    返回：
        int，单位为Byte。