mindspore.ops.Receive
======================

.. py:class:: mindspore.ops.Receive(sr_tag, src_rank, shape, dtype, group="hccl_world_group/nccl_world_group")

    从src_rank接收张量。

    .. note::
        Send和Receive必须组合使用，并且具有相同的sr_tag。Receive必须在服务器之间使用。

    参数：
        - **sr_tag** (int) - 标识发送/接收消息标签的所需的整数。消息将将由具有相同 `sr_tag` 的Send算子发送。
        - **src_rank** (int) - 标识设备rank的所需整数。
        - **shape** (list[int]) - 标识要接收的Tensor的shape的所需列表。
        - **dtype** (Type) - 标识要接收的Tensor类型的必要类型。支持的类型：int8、int16、int32、float16和float32。
        - **group** (str，可选) - 工作通信组。默认值：在Ascend上为“hccl_world_group”，在GPU上为”nccl_world_group”。

    输入：
        - **input_x** (Tensor) - 输入Tensor，其shape为 :math:`(x_1, x_2, ..., x_R)` 。



