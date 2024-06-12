mindspore.hal.reset_max_memory_allocated
========================================

.. py:function:: mindspore.hal.reset_max_memory_allocated(device_target=None)

    重置内存池真实被Tensor占用的内存大小的峰值。

    .. note::
        - 若用户不指定 `device_target` ，将此参数设置为当前已经设置的后端类型。

    参数：
        - **device_target** (str，可选) - 用户指定的后端类型，必须是 ``"CPU"`` ， ``"GPU"`` 以及 ``"Ascend"`` 的其中一个。默认值：``None``。
