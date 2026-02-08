mindspore.mint.distributed.get_rank
=====================================

.. py:function:: mindspore.mint.distributed.get_rank(group=None)

    在指定通信组中获取当前的设备序号。

    .. note::
        - `get_rank` 方法应该在 :func:`mindspore.mint.distributed.init_process_group` 方法之后使用。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **group** (str，可选) - 通信组名称，通常由 :func:`mindspore.mint.distributed.new_group` 方法创建，如果为 ``None`` ， Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None`` 。

    返回：
        int，调用该方法的进程对应的组内序号。

    异常：
        - **TypeError** - 在参数 `group` 不是字符串时抛出。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst
