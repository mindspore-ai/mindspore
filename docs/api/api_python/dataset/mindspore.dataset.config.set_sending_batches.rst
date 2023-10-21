mindspore.dataset.config.set_sending_batches
=============================================

.. py:function:: mindspore.dataset.config.set_sending_batches(batch_num)

    设置Host向Device发送数据的批数上限。

    可用于实现自定义的数据发送控制逻辑，以解决Device内存不足的问题。在每个epoch中，当实际向Device发送的批数达到该值时，
    Host将停止继续发送，直到用户再次通过该接口增大这个上限。

    当前仅支持在Ascend后端的下沉模式训练时使用，下沉模式可通过 :class:`mindspore.train.Model.train` 接口开启。

    参数：
        - **batch_num** (int) - Host向Device发送数据的批数上限。 ``0`` 表示没有发送上限。

    异常：
        - **TypeError** - `batch_num` 不是int类型。
