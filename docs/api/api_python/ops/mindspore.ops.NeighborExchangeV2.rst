mindspore.ops.NeighborExchangeV2
=================================

.. py:class:: mindspore.ops.NeighborExchangeV2(send_rank_ids, recv_rank_ids, send_lens, recv_lens, data_format, group=GlobalComm.WORLD_COMM_GROUP)

    NeighborExchangeV2是一个集合通讯操作。

    将数据从本地rank发送到 `send_rank_ids` 中指定的rank，同时从 `recv_rank_ids` 接收数据。请参考 `分布式集合通信原语 - NeighborExchangeV2 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/communicate_ops.html#neighborexchangev2>`_ 了解具体的数据是如何在相邻设备间交换的。

    .. note::
        要求全连接配网，每台设备具有相同的vlan id，ip和mask在同一子网，请查看 `分布式集合通信原语注意事项 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/communicate_ops.html#注意事项>`_ 。

    参数：
        - **send_rank_ids** (list(int)) - 指定发送数据的rank。8个rank_id分别代表8个方向上的数据要向哪个rank发送，如果某个方向上不发送数据，则设为-1。
        - **recv_rank_ids** (list(int)) - 指定接收数据的rank。8个rank_id分别代表8个方向上的数据要从哪个rank接收，如果某个方向上不接收数据，则设为-1。
        - **send_lens** (list(int)) - 指定 `send_rank_ids` 发送数据的长度，4个数字分别代表[send_top, send_bottom, send_left, send_right]4个方向上的长度。
        - **recv_lens** (list(int)) - 指定 `recv_rank_ids` 接收数据的长度，4个数字分别代表[recv_top, recv_bottom, recv_left, recv_right]4个方向上的长度。
        - **data_format** (str) - 数据格式，现在只支持NCHW。
        - **group** (str, 可选) - 工作的通信组。默认值："GlobalComm.WORLD_COMM_GROUP"（即Ascend平台为"hccl_world_group"，GPU平台为"nccl_world_group" ）。
    输入：
        - **input_x** (Tensor) - 交换前的输入Tensor，其shape为 :math:`(N, C, H, W)`。

    输出：
        数据交换后的输出Tensor，如果输入的shape是 :math:`(N, C, H, W)` ，则输出shape为 :math:`(N, C, H+recv\_top+recv\_bottom, W+recv\_left+recv\_right)` 。

    异常：
        - **TypeError** - 如果 `group` 不是一个string或者 `send_rank_ids` 、 `recv_rank_ids` 、 `send_lens` 和 `recv_lens` 中任意一个不是一个list。
        - **ValueError** - 如果 `send_rank_ids` 或者 `recv_rank_ids` 存在小于-1的值或者存在重复值。
        - **ValueError** - 如果 `send_lens` 或者 `recv_lens` 存在小于零的值。
        - **ValueError** - 如果 `data_format` 不是"NCHW"。

    样例：

    .. note::
        .. include:: mindspore.ops.comm_note.rst

        该样例需要在2卡环境下运行。