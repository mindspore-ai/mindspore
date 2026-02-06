mindspore.communication.get_world_rank_from_group_rank
======================================================

.. py:function:: mindspore.communication.get_world_rank_from_group_rank(group, group_rank_id)

    由指定通信组中的设备序号获取通信集群中的全局设备序号。

    .. note::
        - MindSpore的GPU和CPU版本不支持此方法。
        - 参数 `group` 不能是 ``"hccl_world_group"``。
        - `get_world_rank_from_group_rank` 方法应该在 `init` 方法之后使用。

    参数：
        - **group** (str) - 传入的通信组名称，通常由 `create_group` 方法创建。
        - **group_rank_id** (int) - 通信组内的设备序号。

    返回：
        int，通信集群中的全局设备序号。

    异常：
        - **TypeError** - 参数 `group` 不是字符串，或参数 `group_rank_id` 不是数字。
        - **ValueError** - 参数 `group` 是 ``"hccl_world_group"``，或后端不可用。
        - **RuntimeError** - `HCCL` 服务不可用，或使用了MindSpore的GPU、CPU版本。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst
