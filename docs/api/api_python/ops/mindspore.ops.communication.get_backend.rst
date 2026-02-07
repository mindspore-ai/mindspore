mindspore.ops.communication.get_backend
========================================

.. py:function:: mindspore.ops.communication.get_backend(group=None)

    获取通信进程组的后端。

    .. note::
        MindSpore每个进程仅支持一个通信后端。它应该是 `hccl`/`nccl`/`mccl` 之一。目前仅支持hccl和mccl。

    参数：
        - **group** (str，可选) - 要工作的通信组。通常，组应由 :func:`mindspore.ops.communication.new_group` 创建，默认值： ``None``，在Ascend中表示 ``"hccl_world_group"``。

    返回：
        str，组的后端。

    异常：
        - **TypeError** - 如果 `group` 不是str。

    样例：

    .. note::
        .. include:: ../mint/mindspore.mint.comm_note.rst
