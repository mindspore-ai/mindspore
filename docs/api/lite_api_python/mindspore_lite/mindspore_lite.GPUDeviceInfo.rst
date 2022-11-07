mindspore_lite.GPUDeviceInfo
============================

.. py:class:: mindspore_lite.GPUDeviceInfo(device_id=0, enable_fp16=False)

    用于描述GPU设备硬件信息的辅助类，继承 :class:`mindspore_lite.DeviceInfo` 基类。

    参数：
        - **device_id** (int，可选) - 设备id。默认值：0。
        - **enable_fp16** (bool，可选) - 启用以执行Float16推理。默认值：False。

    异常：
        - **TypeError** - `device_id` 不是int类型。
        - **TypeError** - `enable_fp16` 不是bool类型。
        - **ValueError** - `device_id` 小于0。

    .. py:method:: get_group_size()

        从上下文获取集群数量。

        返回：
            int，集群数量。

    .. py:method:: get_rank_id()

        从上下文获取当前设备在集群中的ID。

        返回：
            int，当前设备在集群中的ID，固定从0开始编号。
