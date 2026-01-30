mindspore.ops.communication.send_object_list
=============================================

.. py:function:: mindspore.ops.communication.send_object_list(object_list, dst, group=None, device=None)

    将输入的Python对象列表同步发送到目的卡上。

    .. note::
        - 当前仅支持PyNative模式，不支持Graph模式。
        - 类似于 :func:`mindspore.ops.communication.send`，但可以传递Python对象。

    .. warning::
        本接口会隐式使用pickle模块，该模块是不安全的。核心原因是反序列化时会执行任意代码，攻击者可通过构造恶意对象触发系统命令执行。因此，调用者须自行保障接口使用数据的安全性。

    参数：
        - **object_list** (list[Any]) - 要发送的输入对象列表。每个对象必须是可pickle的。接收方必须提供大小相等的列表。
        - **dst** (int) - 发送 `object_list` 的目标全局进程组编号。
        - **group** (str，可选) - 通信组名称。如果为 ``None`` ，Ascend平台表示为 ``"hccl_world_group"`` 。默认值： ``None`` 。
        - **device** (str，可选) - 目前这是一个保留参数。默认值： ``None`` 。

    异常：
        - **TypeError** - 如果 `dst` 不是int类型、 `group` 不是str类型，或者 `object_list` 不是list类型。
        - **ValueError** - 如果发送线程和目的线程号相同。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst
