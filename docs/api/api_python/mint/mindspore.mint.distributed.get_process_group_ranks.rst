mindspore.mint.distributed.get_process_group_ranks
======================================================

.. py:function:: mindspore.mint.distributed.get_process_group_ranks(group=None)

    获取指定通信组中的进程，并将通信组中的进程编号以列表方式返回。

    .. note::
        当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **group** (str，可选) - 通信组名称，通常由 :func:`mindspore.mint.distributed.new_group` 方法创建，如果为 ``None`` ， Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None`` 。

    返回：
        List (List[int]) - 指定通信组中的进程编号列表。

    异常：
        - **TypeError** - `group` 不是字符串。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst
