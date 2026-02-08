mindspore.set_device
=====================

.. py:function:: mindspore.set_device(device_target, device_id=None)

    设置运行环境的设备类型和设备ID。

    .. note::
        建议在调用接口 :func:`mindspore.communication.init` 前设置 `device_target` 和 `device_id`。

    参数：
        - **device_target** (str) - 要运行的目标设备，仅支持 ``"Ascend"`` 、 ``"GPU"`` 和 ``"CPU"``。
        - **device_id** (int，可选) - 目标设备的 ID，值必须在 [0, device_num_per_host-1] 范围中，其中 ``device_num_per_host`` 指主机上的设备总数。默认值： ``None``。框架将根据场景按需设置不同的默认值：如果是单卡场景，则设置为 ``0``；如果是msrun启动的分布式场景，则会自动协商可用的 `device_id` 值；如果是其他启动方式的分布式场景，则设置为 ``0``。