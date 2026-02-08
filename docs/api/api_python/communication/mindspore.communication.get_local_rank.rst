mindspore.communication.get_local_rank
======================================

.. py:function:: mindspore.communication.get_local_rank(group=GlobalComm.WORLD_COMM_GROUP)

    获取指定通信组中当前设备的本地设备序号。

    .. note::
        - MindSpore的GPU和CPU版本不支持此方法。
        - `get_local_rank` 方法应该在 `init` 方法之后使用。

    参数：
        - **group** (str, 可选) - 通信组名称，通常由 `create_group` 方法创建，否则将使用默认组名称。默认值：``GlobalComm.WORLD_COMM_GROUP`` 。

    返回：
        int，指定通信组中当前设备的本地设备序号。

    异常：
        - **TypeError** - 参数 `group` 不是字符串。
        - **ValueError** - 后端不可用。
        - **RuntimeError** - `HCCL` 服务不可用，或使用了MindSpore的GPU、CPU版本。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst