mindspore.ops.communication.broadcast_object_list
===================================================

.. py:function:: mindspore.ops.communication.broadcast_object_list(object_list, src=0, group=None, device=None)

    广播整个输入Python对象组。

    .. note::
        - 类似于 :func:`mindspore.ops.communication.broadcast`，但可以传入Python对象。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **object_list** (list[Any]) - 如果src是当前进程的rank，则为要发送的输入列表；否则为用于保存接收数据的列表。
        - **src** (int, 可选) - 指定广播Python对象的进程的rank（全局rank）。只有进程 `src` 会广播Python对象。默认值： ``0`` 。
        - **group** (str，可选) - 通信组名称，如果为 ``None`` ，Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None`` 。
        - **device** (str, 可选) - 当前为预留参数。默认值： ``None``。

    异常：
        - **TypeError** - 如果 `src` 不是整数或 `group` 不是字符串。
        - **RuntimeError** - 如果目标设备无效，或后端无效，或分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
