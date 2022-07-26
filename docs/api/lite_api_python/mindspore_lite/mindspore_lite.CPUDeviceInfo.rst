mindspore_lite.CPUDeviceInfo
============================

.. py:class:: mindspore_lite.CPUDeviceInfo(enable_fp16=False)

    用于设置CPU设备信息的Helper类，继承自DeviceInfo基类。

    参数：
        - **enable_fp16** (bool，可选) - 启用以执行float16推理。默认值：False。

    异常：
        - **TypeError** - `enable_fp16` 不是bool类型。
