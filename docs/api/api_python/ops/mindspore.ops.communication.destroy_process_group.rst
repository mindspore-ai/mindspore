mindspore.ops.communication.destroy_process_group
===================================================

.. py:function:: mindspore.ops.communication.destroy_process_group(group=None)

    销毁用户集合通信组。如果group为None或"hccl_world_group"，则销毁所有组并释放集合通信库。

    .. note::
        - 此方法在GPU和CPU版本的MindSpore中不受支持。
        - 此方法应在 :func:`mindspore.ops.communication.init_process_group` 之后使用。

    参数：
        - **group** (str，可选) - 要工作的通信组。通常，组应由 :func:`mindspore.ops.communication.new_group` 创建。如果为 ``None``，在Ascend中表示 ``"hccl_world_group"``。默认值： ``None``。

    异常：
        - **TypeError** - 如果group不是字符串。
        - **RuntimeError** - 如果HCCL不可用或MindSpore是GPU/CPU版本。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst
