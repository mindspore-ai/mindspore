mindspore.ops.NeighborExchange
===============================

.. py:class:: mindspore.ops.NeighborExchange(send_rank_ids, recv_rank_ids, recv_shapes, send_shapes, recv_type,group=GlobalComm.WORLD_COMM_GROUP)

    NeighborExchange是一个集合通信函数。

    将数据从本地rank发送到 `send_rank_ids` 中指定的rank，同时从 `recv_rank_ids` 接收数据。

    .. note::
        在运行以下示例之前，用户需要预置环境变量，请在 `MindSpore <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.ops.html#通信算子>`_ 的官方网站上查看详细信息。

        要求全连接配网，每台设备具有相同的vlan id，ip和mask在同一子网，请查看 `详细信息 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/communicate_ops.html#注意事项>`_ 。
         
    参数：
        - **send_rank_ids** (list(int)) - 指定发送数据的rank。
        - **recv_rank_ids** (list(int)) - 指定接收数据的rank。
        - **recv_shapes** (tuple(list(int))) - 指定 `recv_rank_ids` 接收数据的shape。
        - **send_shapes** (tuple(list(int))) - 指定 `send_rank_ids` 发送数据的shape。
        - **recv_type** (type) - 指定 `recv_rank_ids` 接收的数据类型。
        - **group** (str) - 要处理的通信范围。默认值："GlobalComm.WORLD_COMM_GROUP"。

    输入：
        - **input_x** (tuple[Tensor]) - shape与参数send_shapes相同。

    输出：
        Tuple tensor，shape与参数recv_shapes相同。
