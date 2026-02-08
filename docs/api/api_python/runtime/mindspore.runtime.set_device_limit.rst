mindspore.runtime.set_device_limit
==================================

.. py:function:: mindspore.runtime.set_device_limit(device, cube_num=-1, vector_num=-1)

    设置指定设备的限制核数。

    .. note::
        - 该接口会同步下发和执行，可能会影响性能。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **device** (int) - 指定的设备 ID。
        - **cube_num** (int，可选) - 设置设备的cube核数。默认值： ``-1``，表示不设置。
        - **vector_num** (int，可选) - 设置设备的vector核数。默认值： ``-1``，表示不设置。
