mindspore.ops.NeighborExchangeV2
=================================

.. py:class:: mindspore.ops.NeighborExchangeV2(send_rank_ids, recv_rank_ids, send_lens, recv_lens, data_format, group=GlobalComm.WORLD_COMM_GROUP)

    NeighborExchangeV2是一个集合通信函数。

    将数据从本地rank发送到 `send_rank_ids` 中指定的rank，同时从 `recv_rank_ids` 接收数据。

    .. note::
        在运行以下示例之前，用户需要预置环境变量，请在 `MindSpore <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.ops.html#通信算子>`_ 的官方网站上查看详细信息。

        要求全连接配网，每台设备具有相同的vlan id，ip和mask在同一子网，请查看 `分布式集合通信原语注意事项 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/communicate_ops.html#注意事项>`_ 。

    参数：
        - **send_rank_ids** (list(int)) - 指定发送数据的rank。8个rank_id分别代表8个方向上的数据要向哪个rank发送，如果某个方向上不发送数据，则设为-1。
        - **recv_rank_ids** (list(int)) - 指定接收数据的rank。8个rank_id分别代表8个方向上的数据要从哪个rank接收，如果某个方向上不接收数据，则设为-1。
        - **send_lens** (list(int)) - 指定 `send_rank_ids` 发送数据的长度，4个数字分别代表[top, bottom, left, right]4个方向上的长度。
        - **recv_lens** (list(int)) - 指定 `recv_rank_ids` 接收数据的长度，4个数字分别代表[top, bottom, left, right]4个方向上的长度。
        - **data_format** (str) - 数据格式，现在只支持NCHW。
        - **group** (str, 可选) - 工作的通信组。默认值："GlobalComm.WORLD_COMM_GROUP"（即Ascend平台为"hccl_world_group"，GPU平台为"nccl_world_group" ）。
