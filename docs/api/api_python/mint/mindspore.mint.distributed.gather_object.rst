mindspore.mint.distributed.gather_object
==========================================

.. py:function:: mindspore.mint.distributed.gather_object(obj, object_gather_list=None, dst=0, group=None)

    对通信组的输入Python对象进行聚合。

    .. note::
        - 类似 :func:`mindspore.mint.distributed.gather` 方法，传入的参数为Python对象。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **obj** (Any) - 输入待聚合的Python对象。
        - **object_gather_list** (list[Any]，可选) - 聚合后的Python对象列表，在 `dst` 设备上需保证输出为通信组的大小。默认值： ``None`` 。
        - **dst** (int，可选) - 表示发送源的进程编号。只有该进程会接收聚合后的张量。默认值： ``0`` 。
        - **group** (str，可选) - 通信组名称，如果为 ``None`` ，Ascend平台表示为 ``"hccl_world_group"`` 。默认值： ``None`` 。

    异常：
        - **TypeError** - `dst` 不是int或 `group` 不是str。
        - **TypeError** - `object_gather_list` 的大小不为通信组大小。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
