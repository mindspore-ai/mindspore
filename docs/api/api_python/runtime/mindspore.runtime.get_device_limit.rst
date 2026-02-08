mindspore.runtime.get_device_limit
==================================

.. py:function:: mindspore.runtime.get_device_limit(device)

    返回指定设备限制核数。

    .. note::
        - 该接口会同步下发和执行，可能会影响性能。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **device** (int) - 指定的设备 ID。

    返回：
        dict，查询到设备上限制的核数信息。

