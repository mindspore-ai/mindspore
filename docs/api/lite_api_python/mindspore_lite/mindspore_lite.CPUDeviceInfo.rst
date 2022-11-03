mindspore_lite.CPUDeviceInfo
============================

.. py:class:: mindspore_lite.CPUDeviceInfo(enable_fp16=False)

    用于描述CPU设备硬件信息的辅助类，继承 :class:`mindspore_lite.DeviceInfo` 基类。

    参数：
        - **enable_fp16** (bool，可选) - 是否启用执行Float16推理。默认值：False。

    异常：
        - **TypeError** - `enable_fp16` 不是bool类型。
