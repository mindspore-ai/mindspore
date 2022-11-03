mindspore_lite.AscendDeviceInfo
===============================

.. py:class:: mindspore_lite.AscendDeviceInfo(device_id=0)

    用于描述Ascend设备硬件信息的辅助类，继承 :class:`mindspore_lite.DeviceInfo` 基类。

    参数：
        - **device_id** (int，可选) - 设备id。默认值：0。

    异常：
        - **TypeError** - `device_id` 不是int类型。
        - **ValueError** - `device_id` 小于0。
