mindspore.hal.max_memory_allocated
==================================

.. py:function:: mindspore.hal.max_memory_allocated(device_target)

    返回从进程启动开始，内存池真实被Tensor占用的内存大小的峰值。

    .. note::
        - 若用户不指定 `device_target` ，将此参数设置为当前已经设置的后端类型。

    参数：
        - **device_target** (str) - 用户指定的后端类型，必须是 ``"CPU"`` ， ``"GPU"`` 以及 ``"Ascend"`` 的其中一个。

    返回：
        int，单位为Byte。
