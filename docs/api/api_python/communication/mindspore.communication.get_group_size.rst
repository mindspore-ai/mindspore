mindspore.communication.get_group_size
======================================

.. py:function:: mindspore.communication.get_group_size(group=GlobalComm.WORLD_COMM_GROUP)

    获取指定通信组实例的rank_size。

    .. note::
        - `get_group_size` 方法应该在 `init` 方法之后使用。

    参数：
        - **group** (str, 可选) - 指定通信组实例（由 `create_group` 方法创建）的名称。默认值：``GlobalComm.WORLD_COMM_GROUP``。

    返回：
        int，指定通信组实例的rank_size。

    异常：
        - **TypeError** - 参数 `group` 不是字符串。
        - **ValueError** - 后端不可用。
        - **RuntimeError** - `HCCL`、`NCCL` 或 `MCCL` 服务不可用。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst
