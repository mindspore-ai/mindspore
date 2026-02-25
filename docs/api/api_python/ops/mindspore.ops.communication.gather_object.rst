mindspore.ops.communication.gather_object
==========================================

.. py:function:: mindspore.ops.communication.gather_object(obj, object_gather_list=None, dst=0, group=None)

    在单个进程中从整个组收集Python对象。

    .. note::
        - 类似于 :func:`mindspore.ops.communication.gather`，但可以传入Python对象。
        - 当前仅支持PyNative模式，不支持Graph模式。

    .. warning::
        本接口会隐式使用pickle模块，该模块是不安全的。核心原因是反序列化时会执行任意代码，攻击者可通过构造恶意对象触发系统命令执行。因此，调用者须自行保障接口使用数据的安全性。

    参数：
        - **obj** (Any) - 要收集的Python对象。
        - **object_gather_list** (list[Any], 可选) - 用于收集数据的相同大小的张量列表。在 `dst` rank上，它应该正确调整为该集合的组大小，并将包含输出。默认值： ``None``。
        - **dst** (int, 可选) - 指定接收张量的进程的rank（全局rank）。只有进程 `dst` 会接收收集的张量。默认值： ``0`` 。
        - **group** (str，可选) - 通信组名称。默认值： ``None`` ，即Ascend平台表示为 ``"hccl_world_group"`` 。

    异常：
        - **TypeError** - 如果 `dst` 不是整数，或 `group` 不是字符串。
        - **TypeError** - 如果 `object_gather_list` 的大小不等于组大小。
        - **RuntimeError** - 如果目标设备无效，或后端无效，或分布式初始化失败。

    样例：

    .. note::
        .. include:: ../mint/mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
