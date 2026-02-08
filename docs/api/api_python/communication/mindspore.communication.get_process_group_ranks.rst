mindspore.communication.get_process_group_ranks
===============================================

.. py:function:: mindspore.communication.get_process_group_ranks(group=GlobalComm.WORLD_COMM_GROUP)

    获取指定通信组中的进程编号，并以列表方式返回。

    参数：
        - **group** (str, 可选) - 通信组名称。默认值：``GlobalComm.WORLD_COMM_GROUP`` （即Ascend平台为 ``"hccl_world_group"``，GPU平台为 ``"nccl_world_group"``）。

    返回：
        List (List[int]) - 指定通信组中的进程编号列表。

    异常：
        - **TypeError** - `group` 不是字符串。
        - **RuntimeError** - 如果目标设备无效、后端无效，或分布式初始化失败。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

        该样例需要在4卡环境下运行。