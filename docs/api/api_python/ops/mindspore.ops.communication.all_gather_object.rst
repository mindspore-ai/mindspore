mindspore.ops.communication.all_gather_object
===============================================

.. py:function:: mindspore.ops.communication.all_gather_object(object_list, obj, group=None)

    在指定通信组中聚合Python对象。

    .. note::
        类似于 :func:`mindspore.ops.communication.all_gather`，但可以传入Python对象。

    .. warning::
        本接口会隐式使用pickle模块，该模块是不安全的。核心原因是反序列化时会执行任意代码，攻击者可通过构造恶意对象触发系统命令执行。因此，调用者须自行保障接口使用数据的安全性。

    参数：
        - **object_list** (list[Any]) - 输出Python对象列表。
        - **obj** (Any) - 从当前进程广播的Python对象。
        - **group** (str，可选) - 通信组名称。默认值： ``None`` ，即Ascend平台表示为 ``"hccl_world_group"`` 。

    异常：
        - **TypeError** - `group` 不是str。
        - **TypeError** - 如果 `object_list` 的大小不等于组大小。
        - **RuntimeError** - 如果目标设备无效，或后端无效，或分布式初始化失败。

    样例：

    .. note::
        .. include:: ../mint/mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
