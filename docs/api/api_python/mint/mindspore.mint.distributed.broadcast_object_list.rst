mindspore.mint.distributed.broadcast_object_list
====================================================

.. py:function:: mindspore.mint.distributed.broadcast_object_list(object_list, src=0, group=None, device=None)

    对输入Python对象进行整组广播。

    .. note::
        - 类似 :func:`mindspore.mint.distributed.broadcast` 方法，传入的参数为Python对象。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **object_list** (list[Any]) - 待广播出去的或接收广播的Python对象。
        - **src** (int，可选) - 表示发送源的进程编号。只有该进程会广播张量。默认值： ``0`` 。
        - **group** (str，可选) - 通信组名称，如果为 ``None`` ，Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None`` 。
        - **device** (str，可选) - 当前为预留参数。默认值： ``None`` 。

    异常：
        - **TypeError** - `src` 不是int或 `group` 不是str。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
