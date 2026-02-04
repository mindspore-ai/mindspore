mindspore.communication.get_comm_name
=====================================

.. py:function:: mindspore.communication.get_comm_name(group=GlobalComm.WORLD_COMM_GROUP)

    获取指定通讯组的通讯器名称。

    .. note::
        - MindSpore的GPU和CPU版本不支持此方法。
        - `get_comm_name` 方法应该在 `init` 方法之后使用。

    参数：
        - **group** (str, 可选) - 传入的通信组名称，通常由 `create_group` 方法创建。默认值：``GlobalComm.WORLD_COMM_GROUP`` 。

    返回：
        str，指定通讯组的通讯器名称。

    异常：
        - **TypeError** - 参数 `group` 不是字符串。
        - **ValueError** - 后端不可用。
        - **RuntimeError** - `HCCL` 服务不可用，或使用了MindSpore的GPU、CPU版本。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

