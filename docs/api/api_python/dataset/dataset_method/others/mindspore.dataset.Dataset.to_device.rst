mindspore.dataset.Dataset.to_device
===================================

.. py:method:: mindspore.dataset.Dataset.to_device(send_epoch_end=True, create_data_info_queue=False)

    将数据从CPU传输到GPU、Ascend或其他设备。

    参数：
        - **send_epoch_end** (bool, 可选) - 是否将epoch结束符 `end_of_sequence` 发送到设备，默认值：True。
        - **create_data_info_queue** (bool, 可选) - 是否创建存储数据类型和shape的队列，默认值：False。

    .. note::
        该接口在将来会被删除或不可见。建议使用 `device_que` 接口。
        如果设备为Ascend，则逐个传输数据。每次数据传输的限制为256M。

    返回：
        TransferDataset，用于传输的数据集对象。

    异常：
        - **RuntimeError** - 如果提供了分布式训练的文件路径但读取失败。
