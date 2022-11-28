mindspore.dataset.config.set_sending_batches
=============================================

.. py:function:: mindspore.dataset.config.set_sending_batches(batch_num)

    在昇腾设备中使用sink_mode=True进行训练时，设置默认的发送批次。

    参数：
        - **batch_num** (int) - 表示总的发送批次。当设置了 `batch_num` 时，它将会等待，除非增加发送批次。默认值为0，表示将发送数据集中的所有批次。

    异常：
        - **TypeError** - `batch_num` 不是int类型。
