mindspore.dataset.Dataset.device_que
====================================

.. py:method:: mindspore.dataset.Dataset.device_que(send_epoch_end=True, create_data_info_queue=False)

    将数据异步传输到Ascend/GPU设备上。

    参数：
        - **send_epoch_end** (bool, 可选) - 数据发送完成后是否发送结束标识到设备上，默认值：True。
        - **create_data_info_queue** (bool, 可选) - 是否创建一个队列，用于存储每条数据的数据类型和shape。默认值：False，不创建。

    .. note::
        如果设备类型为Ascend，数据的特征将被逐一传输。每次传输的数据大小限制为256MB。

    返回：
        Dataset，用于帮助发送数据到设备上的数据集对象。
