mindspore.ops.communication.get_group_rank
============================================

.. py:function:: mindspore.ops.communication.get_group_rank(group, global_rank)

    获取与world通信组中的rank ID对应的指定用户通信组中的rank ID。

    .. note::
        此方法应在 :func:`mindspore.ops.communication.init_process_group` 之后使用。

    参数：
        - **group** (str) - 要工作的通信组。通常，组应由 :func:`mindspore.ops.communication.new_group` 创建。如果为 ``None``，在Ascend中表示 ``"hccl_world_group"``。
        - **global_rank** (int) - world通信组中的rank ID。

    返回：
        int，用户通信组中的rank ID。

    异常：
        - **TypeError** - 如果global_rank不是整数或group不是字符串。
        - **RuntimeError** - 如果设备目标无效，或后端无效，或分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在8卡环境下运行。
