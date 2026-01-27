mindspore.mint.distributed.recv_object_list
===========================================

.. py:function:: mindspore.mint.distributed.recv_object_list(object_list, src=0, group=None, device=None)

    同步接收源的进程的Python对象列表。

    .. note::
        - 当前仅支持PyNative模式，不支持Graph模式。
        - 类似于 :func:`mindspore.mint.distributed.recv`，但可以接收Python对象。

    参数：
        - **object_list** (List[Any]) - 要接收对象的列表。必须提供与发送列表大小相等的列表。
        - **src** (int，可选) - 表示发送源的全局进程编号。默认值： ``0`` 。
        - **group** (str，可选) - 通信组名称。如果为 ``None`` ，Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None`` 。
        - **device** (str，可选) - 目前这是一个保留参数。默认值： ``None`` 。

    返回：
        int。若成功，返回 ``0``。

    异常：
        - **TypeError** - 如果 `src` 不是int类型，或者 `group` 不是str类型，或者 `object_list` 不是list类型。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst