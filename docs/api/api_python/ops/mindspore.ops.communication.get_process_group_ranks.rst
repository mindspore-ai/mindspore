mindspore.ops.communication.get_process_group_ranks
====================================================

.. py:function:: mindspore.ops.communication.get_process_group_ranks(group=None)

    获取特定组的rank，并以列表形式返回通信组中的进程rank。

    参数：
        - **group** (str，可选) - 要工作的通信组。通常，组应由 :func:`mindspore.ops.communication.new_group` 创建。如果为 ``None``，在Ascend中表示 ``"hccl_world_group"``。默认值： ``None``。

    返回：
        List (List[int])，指定通信组中的进程rank列表。

    异常：
        - **TypeError** - 如果 `group` 不是str。
        - **RuntimeError** - 如果设备目标无效，或后端无效，或分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在4卡环境下运行。
