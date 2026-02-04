mindspore.mint.distributed.get_global_rank
=============================================

.. py:function:: mindspore.mint.distributed.get_global_rank(group, group_rank)

    由指定通信组中的设备序号获取通信集群中的全局设备序号。

    .. note::
        - `get_global_rank` 方法应该在 :func:`mindspore.mint.distributed.init_process_group` 方法之后使用。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **group** (str) - 通信组名称，通常由 :func:`mindspore.mint.distributed.new_group` 方法创建，如果为 ``None`` ， Ascend平台表示为 ``"hccl_world_group"`` 。
        - **group_rank** (int) - 通信组内的设备序号。

    返回：
        int，通信集群中的全局设备序号。

    异常：
        - **TypeError** - 参数 `group` 不是字符串。
        - **TypeError** - 参数 `group_rank` 不是数字。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst
