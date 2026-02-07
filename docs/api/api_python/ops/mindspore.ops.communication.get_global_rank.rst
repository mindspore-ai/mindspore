mindspore.ops.communication.get_global_rank
=============================================

.. py:function:: mindspore.ops.communication.get_global_rank(group, group_rank)

    返回与用户组中id为 `group_rank` 的rank对应的world组中的rank id。

    .. note::
        此方法应在 :func:`mindspore.ops.communication.init_process_group` 之后使用。

    参数：
        - **group** (str) - 要工作的通信组。通常，组应由 :func:`mindspore.ops.communication.new_group` 创建。如果为 ``None``，在Ascend中表示 ``"hccl_world_group"``。
        - **group_rank** (int) - 要查询的组rank。

    返回：
        一个整数标量，表示world组中的rank id。

    异常：
        - **TypeError** - 如果 `group` 不是str。
        - **TypeError** - 如果 `group_rank` 不是整数。
        - **RuntimeError** - 如果设备目标无效，或后端无效，或分布式初始化失败。

    样例：

    .. note::
        .. include:: ../mint/mindspore.mint.comm_note.rst

        该样例需要在8卡环境下运行。
