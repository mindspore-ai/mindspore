mindspore.ops.communication.scatter_object_list
================================================

.. py:function:: mindspore.ops.communication.scatter_object_list(scatter_object_output_list, scatter_object_input_list, src=0, group=None)

    将 `scatter_object_input_list` 中的可pickle对象分散到整个组。

    .. note::
        - 类似于 :func:`mindspore.ops.communication.scatter`，但可以传入Python对象。
        - 只有进程 `src` （全局rank）会执行分散操作。
        - 当前仅支持PyNative模式，不支持Graph模式。

    .. warning::
        本接口会隐式使用pickle模块，该模块是不安全的。核心原因是反序列化时会执行任意代码，攻击者可通过构造恶意对象触发系统命令执行。因此，调用者须自行保障接口使用数据的安全性。

    参数：
        - **scatter_object_output_list** (list[Any]) - 非空列表，其第一个元素将存储分散到此rank的对象。
        - **scatter_object_input_list** (list[Any]) - 要分散的Python对象列表。必须在源rank上指定。
        - **src** (int, 可选) - 指定发送张量的进程的rank（全局rank）。只有进程 `src` 会发送张量。默认值： ``0`` 。
        - **group** (str，可选) - 通信组名称。默认值： ``None`` ，即Ascend平台表示为 ``"hccl_world_group"`` 。

    异常：
        - **TypeError** - 如果 `group` 不是str或 `src` 不是整数。
        - **TypeError** - 如果 `scatter_object_input_list` 的大小不等于组大小。
        - **RuntimeError** - 如果目标设备无效，或后端无效，或分布式初始化失败。

    样例：

    .. note::
        .. include:: ../mint/mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
