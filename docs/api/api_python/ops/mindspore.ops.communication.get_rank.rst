mindspore.ops.communication.get_rank
=====================================

.. py:function:: mindspore.ops.communication.get_rank(group=None)

    获取指定集合通信组中当前设备的rank ID。

    .. note::
        此方法应在 :func:`mindspore.ops.communication.init_process_group` 之后使用。

    参数：
        - **group** (str，可选) - 要工作的通信组。通常，组应由 :func:`mindspore.ops.communication.new_group` 创建。默认值： ``None``，在Ascend中表示 ``"hccl_world_group"``。

    返回：
        int，组内调用进程的rank ID。如果不在组内，返回-1。

    异常：
        - **TypeError** - 如果 `group` 不是字符串。

    样例：

    .. note::
        .. include:: ../mint/mindspore.mint.comm_note.rst
