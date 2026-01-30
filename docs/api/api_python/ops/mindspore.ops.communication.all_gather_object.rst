mindspore.ops.communication.all_gather_object
===============================================

.. py:function:: mindspore.ops.communication.all_gather_object(object_list, obj, group=None)

    在指定通信组中聚合Python对象。

    .. note::
        类似于 :func:`mindspore.ops.communication.all_gather`，但可以传入Python对象。

    参数：
        - **object_list** (list[Any]) - 输出Python对象列表。
        - **obj** (Any) - 从当前进程广播的Python对象。
        - **group** (str，可选) - 通信组名称，如果为 ``None`` ，Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None`` 。

    异常：
        - **TypeError** - `group` 不是str。
        - **TypeError** - 如果 `object_list` 的大小不等于组大小。
        - **RuntimeError** - 如果目标设备无效，或后端无效，或分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
