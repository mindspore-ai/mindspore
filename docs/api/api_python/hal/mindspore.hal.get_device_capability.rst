mindspore.hal.get_device_capability
===================================

.. py:function:: mindspore.hal.get_device_capability(device_id, device_target=None)

    返回指定卡号设备的设备能力。

    .. note::
        - 若用户不指定 `device_target` ，将此参数设置为当前已经设置的后端类型。

    参数：
        - **device_id** (int) - 要查询的设备id。
        - **device_target** (str，可选) - 默认值：``None``，必须是 ``"CPU"`` 、 ``"GPU"`` 以及 ``"Ascend"`` 的其中一个。

    返回：
        对于GPU后端，返回tuple(int, int)。

        - param1：int，cuda major版本号。
        - param2：int，cuda minor版本号。

        对于CPU以及Ascend后端，返回None。
