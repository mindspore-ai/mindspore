mindspore.mint.distributed.all_gather_object
===============================================

.. py:function:: mindspore.mint.distributed.all_gather_object(object_list, obj, group=None)

    汇聚指定的通信组中的Python对象。

    .. note::
        - 类似 :func:`mindspore.mint.distributed.all_gather` 方法，传入的参数为Python对象。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **object_list** (list[Any]) - 输出汇聚的Python对象列表。
        - **obj** (Any) - 输入待汇聚操作的Python对象。
        - **group** (str，可选) - 通信组名称，如果为 ``None`` ，Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None`` 。

    异常：
        - **TypeError** - `group` 不是str。
        - **TypeError** - `object_list` 的大小不为通信组大小。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
 